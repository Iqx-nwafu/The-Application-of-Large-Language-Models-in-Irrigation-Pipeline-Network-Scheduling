"""学生模型闭环评测：本地 llama-server（OpenAI 兼容，serve.py 起）当决策器，跑与蒸馏数据同口径的精简工作流——直接调用
distill_gen.run_job：一致性版校核（consistency.install）、无 screen_transient、每轮工具 ≤ 20、K = 3、故障 iso_transient。
每个作业一条 distill-traj/1 记录，写 results/student/<tag>/<task>/<job_id>.json（已存在即跳过，断点续跑；格式与蒸馏原始轨迹相同，
学生自采样 + 仿真拒绝采样时 to_sft.py --src 可直接读）。分组与故障另算一次 GA 参照（缓存 results/student/opt_ref.json，各 checkpoint 共用）。

irrig 环境（要 wntr 与 openai）：
  python src/student/eval_student.py --set val_ckpt --tag v0_s00163_q8 --port 8091 [--lanes 2] [--temperature 0]
  python src/student/eval_student.py --summary v0_s00163_q8 v0_s00326_q8 ...       # 各 tag 的指标对比（--csv 另存）
默认贪心解码（temperature 0、seed 0）；生成提示不加空思考块（enable_thinking=True），见下；每次调用最多生成 4096 token（训练数据单次输出最长约 2.7k），防止复读拖满上下文。
请求超出服务端上下文时记 status=error（error 含 exceed / context），汇总里按兜底前不可行计。
学生是非思考模型，但不能用 enable_thinking=False：那样每次调用的生成提示末尾都有 "<think>

</think>

"，而 Qwen3 模板只给最后一条
assistant 消息渲染这个空思考块——训练数据里它只出现在最终 JSON 前，学生一见就直接交卷、从不调工具、各轮原样重复（11-05 v0 首测踩到）。
enable_thinking=True 时生成提示只到 "<|im_start|>assistant
"，与训练一致：工具调用轮直接写 <tool_call>，交卷轮自己先写空思考块。
"""
import argparse
import collections
import glob
import json
import os
import pathlib
import statistics as st
import subprocess
import sys
import time
import urllib.request

for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):   # 单线程 BLAS，须在 import numpy 前（WMI 起的进程不继承会话环境）
    os.environ.setdefault(k, "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "distill")]
import distill_gen as g                                               # noqa: E402
import consistency as cons                                            # noqa: E402
import evalrun                                                        # noqa: E402
import scoring                                                        # noqa: E402
import teacher                                                        # noqa: E402
import tools                                                          # noqa: E402

OUT = ROOT / "results" / "student"
REF = OUT / "opt_ref.json"
DELTA = json.load(open(ROOT / "thresholds.json", encoding="utf-8"))["delta"]
MAX_TOKENS = 4096


class LocalClient(teacher.Client):
    """llama-server 的 OpenAI 兼容接口；usage 里另记服务端 timings（预填充 / 解码 token 数与毫秒、缓存命中）。"""
    def __init__(self, port, temperature, seed, name):
        from openai import OpenAI
        self.model, self.dry, self.provider, self.temperature, self.seed = name, False, "local", temperature, seed
        self.abort_repeats = 0
        self.cli = OpenAI(api_key="local", base_url=f"http://127.0.0.1:{port}/v1", timeout=teacher.REQUEST_TIMEOUT_S, max_retries=0)

    def chat(self, messages, tools_spec):
        import openai
        t0 = time.time()
        self.last = messages
        try:
            r = self.cli.chat.completions.create(model=self.model, messages=messages, tools=[{"type": "function", "function": s} for s in tools_spec],
                                                 temperature=self.temperature, seed=self.seed, max_tokens=MAX_TOKENS,
                                                 extra_body=dict(chat_template_kwargs=dict(enable_thinking=True)))
        except openai.InternalServerError as e:
            if "parse tool call" not in str(e):
                raise
            # 服务端解析不了模型写的 <tool_call>（参数不是合法 JSON，常见于被 MAX_TOKENS 截断的复读）：API 教师同类错误会作为工具报错反馈，
            # 这里当成一次格式错误的作答——solve 解析不出 JSON，记 format_error 并把反馈发回，学生下一轮重来
            return (dict(role="assistant", content=f"[tool_call 解析失败] {str(e)[:200]}"),
                    dict(prompt_tokens=None, completion_tokens=None, retry_wait_s=0.0, tool_parse_error=True), time.time() - t0)
        u = r.usage.model_dump() if r.usage else {}
        u.update(served_model=r.model, retry_wait_s=0.0, finish_reason=r.choices[0].finish_reason, timings=(r.model_extra or {}).get("timings"))
        reply = r.choices[0].message.model_dump(exclude_none=True)
        if self.abort_repeats and reply.get("tool_calls"):                 # 自采样提速：连续第 n 次一字不差的调用就判失败（11-08 查明失败轨迹都是这样复读到撑满）
            sig = lambda m: [(c["function"]["name"], c["function"]["arguments"]) for c in m.get("tool_calls") or []]
            prev = [sig(m) for m in messages if m["role"] == "assistant" and m.get("tool_calls")]
            k = next((i for i, s in enumerate(reversed(prev)) if s != sig(reply)), len(prev))
            if k + 1 >= self.abort_repeats:
                raise RuntimeError(f"abort: {k + 1} 次连续相同的工具调用")
        return reply, u, time.time() - t0


def configure(max_calls=20):
    """与 distill_gen main() 的新项目默认口径相同（--consistency on --tools lean --max-tool-calls 20 --iso-transient on）；
    max_calls：每轮工具上限（学生另测 12 一组，与第一轮 API 结果同口径）。"""
    cons.install()
    g.CHECKER, g.ISO_TRANSIENT, g.MAX_CALLS = "v2-consistency", True, max_calls
    g.TOOL_SPECS = [t for t in tools.TOOL_SPECS if t["name"] != "screen_transient"]


def state_key(job):
    v = job["variant"]
    return f"{job['task']}|{job['scenario']}|{v.get('pair', v.get('t', ''))}"


def get_ref(job, S):
    """分组 / 故障状态的 GA 参照（同 to_sft.compute_refs：GA 方案本身的 C* 与缺水），缺就算一次并缓存。"""
    if job["task"] == "switch":
        return None
    k = state_key(job)
    ref = json.load(open(REF, encoding="utf-8")) if REF.exists() else {}
    if k not in ref:
        sc = g.scenario_for(job, S)
        ctx, _ = teacher.make_state(job["task"], sc, g.rng_for(job))
        if not ctx.valves:
            return None
        o = tools.run_optimizer(ctx, sc, apply_faults=(job["task"] == "fault_response"))
        ref = evalrun.save_ref(REF, k, dict(score=o["score"], deficit=o["objectives"]["deficit_m3"], feasible=o["feasible"]))
    return ref[k]


def server_info(port):
    try:
        p = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=10))
        return dict(model_path=p.get("model_path"), n_ctx=(p.get("default_generation_settings") or {}).get("n_ctx"), build=p.get("build_info"))
    except OSError as e:
        raise SystemExit(f"llama-server 127.0.0.1:{port} 连不上：{e}")


def worker(a, jobs):
    si, sk = map(int, a.shard.split("/"))
    configure(a.max_tool_calls)
    S = {s["id"]: s for s in (json.loads(l) for l in open(ROOT / "scenarios.jsonl", encoding="utf-8"))}
    info = server_info(a.port)
    client = LocalClient(a.port, a.temperature, a.seed, a.tag)
    client.abort_repeats = a.abort_repeats
    for job in jobs[si::sk]:
        p = OUT / a.tag / job["task"] / f"{job['job_id']}.json"
        if p.exists():
            continue
        t0 = time.time()
        client.last, client.seed = None, a.seed + job["sample"]               # 自采样：同一状态的各次采样用不同随机种子
        rec = g.run_job(client, job, S, "local")
        if rec["status"] != "ok" and client.last:                       # 出错（多为超上下文）时留下出错前的对话，便于看学生卡在哪
            rec["partial_messages"] = client.last
        rec.update(eval=dict(tag=a.tag, set=a.set, port=a.port, temperature=a.temperature, seed=client.seed, max_tokens=MAX_TOKENS,
                            max_tool_calls=a.max_tool_calls, abort_repeats=a.abort_repeats, server=info))
        rec["ref"] = None if a.no_ref else get_ref(job, S)               # 自采样不算：to_sft 用轨迹里的 GA 结果与 distill/ref*.json，同状态 4 次采样并行时还会重复跑 GA
        g.atomic_write(p, rec)
        o = rec.get("outcome") or {}
        print(f"[s{si}] {job['job_id']} {rec['status']} 兜底前可行 {o.get('feasible_before_fallback')} 一致 {(o.get('consistency') or {}).get('consistent')} "
              f"缺水 {(o.get('objectives') or {}).get('deficit_m3')} C* {o.get('score')} 参照 {rec['ref']} 轮 {o.get('n_rounds')} 工具 {o.get('tool_calls')} "
              f"模型 {(rec.get('latency') or {}).get('model_s')} s 共 {time.time() - t0:.0f} s {rec.get('error', '')}", flush=True)


# ---------------- 汇总 ----------------
def row_metrics(r):
    """一条记录 → 指标。达标 = 兜底前可行 + 隔离与计划一致 + 缺水 ≤ GA + d_tol + C* 差距 ≤ δ（同 to_sft 的可用口径；切换只看可行）。"""
    o, ref = r.get("outcome") or {}, r.get("ref")
    ok = r["status"] == "ok"
    bf = ok and bool(o.get("feasible_before_fallback"))
    cons_ok = r["task"] != "fault_response" or (ok and bool((o.get("consistency") or {}).get("consistent")))
    obj = o.get("objectives") or {}
    dfc, score = obj.get("deficit_m3"), o.get("score")
    tol = scoring.d_tol(DT[r["scenario"]["base_id"]])
    d_ex = None if (dfc is None or not ref) else dfc - ref["deficit"]
    gap = None if (score is None or not ref or ref.get("score") is None) else ref["score"] - score
    good = bf and cons_ok and (r["task"] == "switch" or (d_ex is not None and d_ex <= tol and gap is not None and gap <= DELTA))
    rounds = r.get("rounds") or []
    usage = [m.get("_usage") or {} for m in r.get("messages") or [] if m["role"] == "assistant"]
    ctx_full = any(u.get("finish_reason") == "length" and (u.get("completion_tokens") or 0) < MAX_TOKENS for u in usage)   # 生成被上下文顶满截断
    return dict(task=r["task"], fault=(r["scenario"].get("fault") or {}).get("type"), ok=ok,
                ctx_err=(not ok and "context" in (r.get("error") or "").lower()) or ctx_full,
                bf=bf, cons=cons_ok, final=ok and bool(o.get("final_feasible")), good=good,
                d_ex=d_ex if bf else None, gap=gap if bf else None,
                schema=[("format_error" not in x["check"]) for x in rounds], first_schema=bool(rounds) and "format_error" not in rounds[0]["check"],
                tool_calls=o.get("tool_calls"), n_rounds=o.get("n_rounds"), cap=bool(o.get("tool_cap_hit")),
                model_s=(r.get("latency") or {}).get("model_s"), wall_s=(r.get("latency") or {}).get("wall_s"),
                in_window=bool((r.get("latency") or {}).get("in_window")),
                out_tok=sum(u.get("completion_tokens") or 0 for u in usage),
                decode=[u["timings"]["predicted_per_second"] for u in usage if (u.get("timings") or {}).get("predicted_per_second")],
                prefill=[u["timings"]["prompt_per_second"] for u in usage if (u.get("timings") or {}).get("prompt_n", 0) >= 512])


def med(v):
    v = [x for x in v if x is not None]
    return round(st.median(v), 3) if v else None


def summarize(tags, csv_path=None):
    lines = []
    for tag in tags:
        R = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(str(OUT / tag / "*" / "*.json")))]
        M = [row_metrics(r) for r in R if r["status"] != "degenerate"]
        for key in ("switch", "rotation_plan", "fault_response", "fault_burst", "fault_leak", "fault_stuck", "all"):
            ms = [m for m in M if key == "all" or m["task"] == key or (key.startswith("fault_") and m["fault"] == key[6:])]
            if not ms:
                continue
            n = len(ms)
            sch = [x for m in ms for x in m["schema"]]
            lines.append(dict(tag=tag, group=key, n=n, error=sum(not m["ok"] for m in ms), ctx_err=sum(m["ctx_err"] for m in ms),
                              feasible_bf=round(sum(m["bf"] for m in ms) / n, 3), consistent=round(sum(m["cons"] for m in ms) / n, 3),
                              final_feasible=round(sum(m["final"] for m in ms) / n, 3), good=round(sum(m["good"] for m in ms) / n, 3),
                              deficit_excess_med=med(m["d_ex"] for m in ms), gap_med=med(m["gap"] for m in ms),
                              schema_ok=round(sum(sch) / len(sch), 3) if sch else None, first_schema_ok=round(sum(m["first_schema"] for m in ms) / n, 3),
                              tool_calls_med=med(m["tool_calls"] for m in ms), rounds_mean=round(sum(m["n_rounds"] or 0 for m in ms) / n, 2),
                              cap_hit=round(sum(m["cap"] for m in ms) / n, 3), model_s_med=med(m["model_s"] for m in ms),
                              wall_s_med=med(m["wall_s"] for m in ms), in_window=round(sum(m["in_window"] for m in ms) / n, 3),
                              out_tok_med=med(m["out_tok"] for m in ms), decode_tok_s_med=med(x for m in ms for x in m["decode"]),
                              prefill_tok_s_med=med(x for m in ms for x in m["prefill"])))
    cols = ["tag", "group", "n", "error", "ctx_err", "feasible_bf", "consistent", "final_feasible", "good", "deficit_excess_med", "gap_med",
            "schema_ok", "first_schema_ok", "tool_calls_med", "rounds_mean", "cap_hit", "model_s_med", "wall_s_med", "in_window", "out_tok_med", "decode_tok_s_med", "prefill_tok_s_med"]
    print("\t".join(cols))
    for l in lines:
        print("\t".join(str(l[c]) for c in cols))
    if csv_path:
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(lines)


DT = {s["id"]: sum(s["demand_m3"].values()) for s in (json.loads(l) for l in open(ROOT / "scenarios.jsonl", encoding="utf-8"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="val_ckpt", help="student/sets/<名>.json（make_val_set.py）")
    ap.add_argument("--tag", default=None, help="结果子目录名：建议 <版本>_<checkpoint>_<精度>，如 v0_s00163_q8")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tool-calls", type=int, default=20, help="每轮工具调用上限（蒸馏与学生评测 20；另测 12）")
    ap.add_argument("--abort-repeats", type=int, default=0, help="自采样用：连续 n 次完全相同的工具调用即判失败（评测保持 0 = 不启用）")
    ap.add_argument("--no-ref", action="store_true", help="自采样用：不算 GA 参照")
    ap.add_argument("--lanes", type=int, default=1, help="并行路数（共用一个 llama-server，GA 并行、模型调用在服务端排队；正式时延只用 1 路）")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--summary", nargs="+", default=None, help="汇总这些 tag")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    if a.summary:
        return summarize(a.summary, a.csv)
    jobs = json.load(open(HERE / "sets" / f"{a.set}.json", encoding="utf-8"))["jobs"]
    n, seen, keyed = collections.Counter(j["task"] for j in jobs), collections.Counter(), []
    for j in jobs:                                                     # 按任务内相对位置交错（同 distill_gen.load_plan），中途看结果三类都有
        keyed.append(((seen[j["task"]] + 0.5) / n[j["task"]], j["job_id"], j))
        seen[j["task"]] += 1
    jobs = [j for *_, j in sorted(keyed, key=lambda t: t[:2])]
    if a.shard:
        return worker(a, jobs)
    server_info(a.port)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    procs = []
    for i in range(a.lanes):
        log = open(OUT / "logs" / f"{a.tag}.s{i}.log", "a", encoding="utf-8")
        procs.append(subprocess.Popen([sys.executable, __file__, *sys.argv[1:], "--shard", f"{i}/{a.lanes}"], cwd=ROOT, stdout=log,
                                      stderr=subprocess.STDOUT, env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)))
    print(f"{a.lanes} 路已启动，日志 results/student/logs/{a.tag}.s*.log；退出码", [p.wait() for p in procs], flush=True)


if __name__ == "__main__":
    main()
