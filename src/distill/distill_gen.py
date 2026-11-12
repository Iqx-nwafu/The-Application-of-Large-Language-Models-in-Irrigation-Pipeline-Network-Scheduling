"""蒸馏轨迹生成：DeepSeek V4-Pro 教师按 plan 逐 job 跑工作流（teacher.solve，与正式实验 ds_test 同提示、同工具、同 K、同温度），
每条轨迹落一个文件 distill/raw/<task>/<job_id>.json（原子写），已存在就跳过，所以掉线后原命令重跑即续跑。

  python distill/distill_gen.py --plan plan_10000 --budget-cny 3000 --lanes 4          # 前台起 4 路子进程（调试用）
  pythonw distill/launch_distill.py 10000 --plan plan_10000 --budget-cny 3000 --lanes 4   # 正式：无控制台后台（见 launch_distill.py）
  python distill/distill_gen.py --plan plan_10000 --status                              # 进度、累计费用、按任务的可行率
  python distill/distill_gen.py --plan plan_5000 --dry-run --limit 3 --task switch      # 规则桩跑通流程，不调 API
  python distill/distill_gen.py --plan plan_10000 --first 100 --budget-cny 40 --lanes 4  # 试点：执行顺序前 100 条（切换 40、分组 20、故障 40）
  python distill/distill_gen.py --plan plan_10000 --ids distill/plans/pilot40_fault.txt --model deepseek-chat --raw raw_flash --budget-cny 8
                                                                                         # 只跑清单里的 job；Flash 写 raw_flash/
  python distill/distill_gen.py --plan plan_10000 --cascade-from raw_flash --budget-cny ...  # 级联第二级：只重做 raw_flash 里没通过的 job（Pro 写 raw/）

新项目口径（2026-11-02 起默认）：每轮工具上限 20（--max-tool-calls）；精简工具集，去掉 screen_transient（--tools lean）；
故障场景用真实关阀口径 iso_transient（--iso-transient on）；隔离后无可调度支管的退化故障场景不调 API，记 status=degenerate。
--tools full --max-tool-calls 12 --iso-transient off 即试点 100 条的口径。

省钱与安全
  --budget-cny 必填（dry-run 除外）：每个 job 开始前汇总所有路的账本 distill/ledger.s*.jsonl，累计 ≥ 上限就停，在途 job 照常跑完。
  空闲时段门禁：distill/offpeak_only.txt 为 1（默认建）时只在 DeepSeek 空闲时段发起新 job，在场景边界等待，等待不计时延；改 0 临时解除。
  timeout / error 也落文件（status 字段）且不自动重试，避免同一 job 反复花钱；--retry-failed 才重跑。
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import numpy as np

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import teacher                                               # noqa: E402
import evalrun                                               # noqa: E402
import tools                                                 # noqa: E402
import consistency as cons                                   # noqa: E402  （distill/ 下）

SCHEMA_VERSION = "distill-traj/1"
RAW = HERE / "raw"
LEDGER = "ledger"
CHECKER, ISO_TRANSIENT, MAX_CALLS, TOOL_SPECS = "v2-consistency", True, 20, None                # main() 按参数设置
GATE = HERE / "offpeak_only.txt"


class FixedPair:
    """teacher.make_state 对 switch 只调一次 rng.integers(29)；用它把组对固定成 plan 里的那个。"""
    def __init__(self, k):
        self.k = k

    def integers(self, n):
        assert self.k < n
        return self.k


def load_plan(name):
    """按任务内相对位置交错排序：任何时刻停下（预算到顶、中断），已完成部分的任务配比都接近 plan。"""
    p = pathlib.Path(name)
    p = p if p.exists() else HERE / "plans" / f"{name.removesuffix('.jsonl')}.jsonl"
    jobs = [json.loads(l) for l in open(p, encoding="utf-8")]
    n, seen = collections.Counter(j["task"] for j in jobs), collections.Counter()
    keyed = []
    for j in jobs:
        keyed.append(((seen[j["task"]] + 0.5) / n[j["task"]], j["job_id"], j))
        seen[j["task"]] += 1
    return [j for *_, j in sorted(keyed, key=lambda t: t[:2])]


def scenario_for(job, by_id):
    """故障时刻变体：复制场景、改写故障时刻字段（只影响 make_state 的已灌组与剩余时间窗）。"""
    sc = by_id[job["scenario"]]
    timing = job["variant"].get("timing")
    if job["task"] != "fault_response":
        return sc
    sc = json.loads(json.dumps(sc))
    if ISO_TRANSIENT:
        sc["iso_transient"] = True                                    # 隔离关阀计入瞬态（只对爆管的 isolate_valves 起作用）
    if not timing:
        return sc
    f = sc["faults"][0]
    for k in ("phase", "cycle_frac", "switch_frac", "t_in_switch_s"):
        f.pop(k, None)
    f.update(timing)
    sc["id"] = f"{sc['id']}~t{job['variant']['t']}"
    return sc


def rng_for(job):
    if job["task"] == "switch":
        return FixedPair(job["variant"]["pair"])
    return np.random.default_rng([job["seed"], int(job["scenario"][1:]), job["sample"]])


def out_path(job):
    return RAW / job["task"] / f"{job['job_id']}.json"


def atomic_write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, default=float)
    os.replace(tmp, path)


def spent():
    tot = 0.0
    for fn in glob.glob(str(HERE / f"{LEDGER}.s*.jsonl")):
        for l in open(fn, encoding="utf-8"):
            try:
                tot += json.loads(l).get("cost_cny") or 0.0
            except json.JSONDecodeError:                                 # 别的路正在写的半行
                pass
    return tot


def is_peak_bj(t=None):
    """DeepSeek 高峰：北京时间工作日 9–12、14–18。显式按 UTC+8 算，不用本机时区
    （2026-10-28 发现本机 time.localtime 比北京时间快 5 h，evalrun.is_peak 的门禁因此整体错位）。"""
    lt = time.gmtime((time.time() if t is None else t) + 8 * 3600)
    return lt.tm_wday < 5 and (9 <= lt.tm_hour < 12 or 14 <= lt.tm_hour < 18)


def wait_offpeak():
    t0 = time.time()
    while is_peak_bj():
        try:
            if GATE.read_text().strip() != "1":
                break
        except OSError:
            break
        time.sleep(60)
    return round(time.time() - t0, 1)


def ga_scores(msgs):
    """轨迹里 run_optimizer 返回的 C*（教师自己调过优化器时，就是这个状态的 GA 参照，不用另算）。"""
    names = {c["id"]: c["function"]["name"] for m in msgs if m["role"] == "assistant" for c in m.get("tool_calls") or []}
    out = []
    for m in msgs:
        if m["role"] == "tool" and names.get(m.get("tool_call_id")) == "run_optimizer":
            try:
                r = json.loads(m["content"])
                if r.get("feasible") and r.get("score") is not None:
                    out.append(r["score"])
            except (json.JSONDecodeError, AttributeError):
                pass
    return out


def scenario_meta(sc, job):
    f = (sc.get("faults") or [None])[0]
    return dict(id=sc["id"], base_id=job["scenario"], split=sc["split"], kind=sc["kind"], irrigation_event=sc["irrigation_event"],
                quota=sc["quota_m3_per_mu"], delta=sc["perturbation"]["delta"],
                fault=None if f is None else {k: f.get(k) for k in ("type", "pipe", "pipe_class", "valve", "mode", "submain",
                                                                        "phase", "cycle_frac", "switch_frac")})


def run_job(client, job, by_id, provider):
    sc = scenario_for(job, by_id)
    offpeak_wait = wait_offpeak() if provider == "deepseek" else 0.0
    rec = dict(schema=SCHEMA_VERSION, job_id=job["job_id"], plan_seed=job["seed"], task=job["task"], variant=job["variant"],
               sample=job["sample"], created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
               teacher=dict(provider=provider, model=client.model, temperature=client.temperature),
               scenario=scenario_meta(sc, job), window_s=teacher.WINDOW_S[job["task"]], checker=CHECKER)
    rec["config"] = dict(max_tool_calls=MAX_CALLS, tools=[t["name"] for t in TOOL_SPECS], iso_transient=bool(sc.get("iso_transient")))
    if job["task"] == "fault_response" and not teacher.make_state(job["task"], sc, rng_for(job))[0].valves:
        rec.update(status="degenerate", error="故障隔离后无可调度支管", cost=None, latency=dict(wall_s=0.0, offpeak_wait_s=offpeak_wait))
        return rec                                                     # 退化场景（同 O00237）：不调 API
    t0 = time.time()
    try:
        log = teacher.solve(client, job["task"], sc, rng_for(job), tool_specs=TOOL_SPECS, max_tool_calls=MAX_CALLS)
    except Exception as e:                                              # 超时 / 重试耗尽：记一条，不自动重跑
        rec.update(status="timeout" if "Timeout" in type(e).__name__ else "error", error=f"{type(e).__name__}: {str(e)[:300]}",
                   latency=dict(wall_s=round(time.time() - t0, 1), offpeak_wait_s=offpeak_wait), cost=None)
        return rec
    msgs = log["messages"]
    asst = [m for m in msgs if m["role"] == "assistant"]
    for m, u in zip(asst, log["usage"]):                               # 每次调用的 usage 挂在对应 assistant 消息上
        m["_usage"] = u
    ctx, state = teacher.make_state(job["task"], sc, rng_for(job))     # 与 solve 内部同一状态（rng 确定）
    tok, cost = evalrun.tokens_cost(log["usage"], client.model)
    if cost["cost_cny"] is not None:                                   # 按北京时间重定计价时段（同 is_peak_bj）
        cost["peak"] = is_peak_bj()
        cost["cost_cny"] = cost["cost_peak_cny"] if cost["peak"] else cost["cost_offpeak_cny"]
    final = evalrun.final_check(ctx, job["task"], log["final_decision"], state)
    cap_hit = any(m["role"] == "tool" and "工具调用已达上限" in m["content"] for m in msgs)
    tool_err = sum(m["role"] == "tool" and m["content"].startswith('{"error"') for m in msgs)
    rec.update(status="ok", state=log["state"],
               prompt=dict(system=msgs[0]["content"], user=msgs[1]["content"],
                           system_sha1=hashlib.sha1(msgs[0]["content"].encode()).hexdigest()[:12]),
               tools=TOOL_SPECS, messages=msgs, rounds=log["rounds"],
               outcome=dict(feasible_before_fallback=log["feasible_before_fallback"], n_rounds=log["n_rounds"],
                            tool_calls=log["tool_calls"], tool_cap_hit=cap_hit, tool_errors=tool_err,
                            format_errors=sum(1 for r in log["rounds"] if "format_error" in r["check"]),
                            fallback_used=log["fallback_used"], decision=log["decision"], final_decision=log["final_decision"],
                            ga_scores=ga_scores(msgs), **final,
                            consistency=cons.check(log["decision"], sc["faults"][0], state["done"])
                            if job["task"] == "fault_response" and log["decision"] else None),
               cost=dict(tokens=tok, **cost),
               latency=dict(model_s=round(log["latency_s"], 1), wall_s=round(log["wall_s"], 1),
                            first_feasible_s=None if log["first_feasible_s"] is None else round(log["first_feasible_s"], 1),
                            retry_wait_s=sum(u.get("retry_wait_s") or 0 for u in log["usage"]), in_window=log["in_window"],
                            offpeak_wait_s=offpeak_wait),
               served_model=next((u.get("served_model") for u in log["usage"] if u.get("served_model")), None))
    return rec


def passed(p):
    """级联判定：兜底前可行，且故障方案隔离与计划一致（R8）。"""
    if not p.exists():
        return False
    r = json.load(open(p, encoding="utf-8"))
    o = r.get("outcome") or {}
    return r.get("status") in ("ok", "degenerate") and (r["status"] == "degenerate" or
        (bool(o.get("feasible_before_fallback")) and (o.get("consistency") or {}).get("consistent", True)))


def worker(a, jobs):
    si, sk = map(int, a.shard.split("/"))
    S = {s["id"]: s for f in (ROOT / "scenarios.jsonl", HERE / "scenarios_stress.jsonl") if f.exists()      # 后者：训练用切换应力场景（make_stress.py）
         for s in (json.loads(l) for l in open(f, encoding="utf-8"))}
    client = teacher.Client(a.model, a.provider, None, a.dry_run)
    ledger = open(HERE / f"{LEDGER}.s{si}.jsonl", "a", encoding="utf-8")
    for job in jobs[si::sk]:
        p = out_path(job)
        if p.exists():
            if not a.retry_failed or json.load(open(p, encoding="utf-8")).get("status") == "ok":
                continue
        if not a.dry_run and spent() >= a.budget_cny:
            print(f"[s{si}] 累计 {spent():.1f} 元 ≥ 上限 {a.budget_cny}，停止", flush=True)
            break
        rec = run_job(client, job, S, a.provider)
        atomic_write(p, rec)
        c = (rec.get("cost") or {}).get("cost_cny")
        ledger.write(json.dumps(dict(job_id=job["job_id"], t=rec["created_at"], status=rec["status"], cost_cny=c)) + "\n"); ledger.flush()
        o = rec.get("outcome") or {}
        if rec["status"] == "degenerate":
            print(f"[s{si}] {job['job_id']} 退化，跳过", flush=True); continue
        print(f"[s{si}] {job['job_id']} {rec['status']} 兜底前可行 {o.get('feasible_before_fallback')} 最终 {o.get('final_feasible')} "
              f"轮 {o.get('n_rounds')} 工具 {o.get('tool_calls')} C* {o.get('score')} {rec['latency']['wall_s']} s 费用 {c}", flush=True)


def status(jobs):
    by_task = collections.defaultdict(collections.Counter)
    cost, wall = collections.defaultdict(float), collections.defaultdict(float)
    for j in jobs:
        p, c = out_path(j), by_task[j["task"]]
        c["plan"] += 1
        if not p.exists():
            continue
        r = json.load(open(p, encoding="utf-8"))
        c[r["status"]] += 1
        if r["status"] == "ok":
            o = r["outcome"]
            c["feasible_bf"] += bool(o["feasible_before_fallback"])
            c["final_feasible"] += bool(o.get("final_feasible"))
            cost[j["task"]] += r["cost"]["cost_cny"] or 0
            wall[j["task"]] += r["latency"]["wall_s"]
    for t, c in by_task.items():
        n = c["ok"] or 1
        print(f"{t:<15} 计划 {c['plan']:>5}  完成 {c['ok']:>5}  超时 {c['timeout']}  错误 {c['error']}  兜底前可行 {c['feasible_bf']}  "
              f"费用 {cost[t]:.1f} 元（{cost[t] / n:.3f}/条）  平均 {wall[t] / n:.0f} s/条")
    print(f"账本累计 {spent():.1f} 元")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--shard", default=None, help="i/k：子进程用，只跑 job 序号 mod k == i")
    ap.add_argument("--budget-cny", type=float, default=None)
    ap.add_argument("--task", nargs="*", default=None, help="只跑这些任务")
    ap.add_argument("--limit", type=int, default=None, help="每个任务只取前 n 个 job（试跑）")
    ap.add_argument("--first", type=int, default=None, help="只跑执行顺序上的前 n 个 job（试点；与正式档位共用，跑过的不重跑）")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--max-tool-calls", type=int, default=20, help="每轮工具调用上限（主实验 12；新项目 20，作者 2026-10-29 定）")
    ap.add_argument("--tools", choices=["lean", "full"], default="lean", help="lean：去掉 screen_transient（作者 2026-11-01 定）")
    ap.add_argument("--iso-transient", choices=["on", "off"], default="on", help="故障场景隔离关阀计入瞬态")
    ap.add_argument("--raw", default="raw", help="轨迹目录名（distill/ 下）；账本随之为 ledger_<raw>.s*.jsonl（raw 用 ledger.s*）")
    ap.add_argument("--ledger", default=None, help="账本名（distill/<名>.s*.jsonl，也进日志名）；同一 --raw 并行起两个启动器时各给一个，"
                                                  "免得两进程同写一个账本；--budget-cny 只按本账本计")
    ap.add_argument("--ids", default=None, help="job_id 清单文件（一行一个），只跑这些")
    ap.add_argument("--cascade-from", default=None, help="级联：跳过该目录里已通过（兜底前可行且一致）的 job")
    ap.add_argument("--consistency", choices=["on", "off"], default="on",
                    help="on：故障处置用一致性版校核（consistency.install：稳态按声明的隔离算、一致性反馈、隔离动作可为空、提示补规则与管段编号）；"
                         "off：与正式实验 ds_test 完全相同")
    a = ap.parse_args()
    global CHECKER, RAW, LEDGER, ISO_TRANSIENT, MAX_CALLS, TOOL_SPECS
    CHECKER = "v2-consistency" if a.consistency == "on" else "v1"
    if a.consistency == "on":
        cons.install()
    ISO_TRANSIENT, MAX_CALLS = a.iso_transient == "on", a.max_tool_calls
    TOOL_SPECS = [t for t in tools.TOOL_SPECS if a.tools == "full" or t["name"] != "screen_transient"]
    RAW, LEDGER = HERE / a.raw, a.ledger or ("ledger" if a.raw == "raw" else f"ledger_{a.raw}")
    jobs = load_plan(a.plan)
    if a.ids:
        keep = set(open(a.ids, encoding="utf-8").read().split())
        jobs = [j for j in jobs if j["job_id"] in keep]
    if a.cascade_from:
        jobs = [j for j in jobs if not passed(HERE / a.cascade_from / j["task"] / f"{j['job_id']}.json")]
    if a.task:
        jobs = [j for j in jobs if j["task"] in a.task]
    if a.limit:
        seen = collections.Counter()
        jobs = [j for j in jobs if (seen.update([j["task"]]) or seen[j["task"]] <= a.limit)]
    if a.first:
        jobs = jobs[:a.first]
    if a.status:
        return status(jobs)
    if a.budget_cny is None and not a.dry_run:
        raise SystemExit("真实生成必须给 --budget-cny（费用上限，元）")
    if a.budget_cny is None:
        a.budget_cny = 0.0
    if a.dry_run:
        RAW, LEDGER = HERE / f"{a.raw}_dry", f"{LEDGER}_dry"
    if not GATE.exists():
        GATE.write_text("1")
    if a.shard:
        return worker(a, jobs)
    (HERE / "logs").mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    procs = []
    for i in range(a.lanes):
        cmd = [sys.executable, __file__, *sys.argv[1:], "--shard", f"{i}/{a.lanes}"]
        log = open(HERE / "logs" / f"gen_{pathlib.Path(a.plan).stem}_{a.ledger or a.raw}.s{i}.log", "a", encoding="utf-8")
        procs.append(subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, creationflags=flags))
    print(f"{a.lanes} 路已启动，日志 distill/logs/；退出码", [p.wait() for p in procs], flush=True)


if __name__ == "__main__":
    main()
