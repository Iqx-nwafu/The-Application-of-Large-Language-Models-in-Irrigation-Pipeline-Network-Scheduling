"""A9 评测框架：同一批场景上跑四类决策器，记录成本、调度结果、响应速度三维指标，写 results/<tag>.jsonl；stats.py 做统计。

决策器（--maker）
  workflow     LLM + 工具 + K 轮反馈 + 兜底（teacher.solve）
  prompt_only  LLM 约束增强提示直出：无工具、无反馈，一次输出后全量校核（复现 Chen 等 2026 范式）
  optimizer    NSGA-II（rotation_plan / fault_response，故障隔离按规则）；switch 用规则切换 + 全量校核逐档放慢（safe_switch）
  rules        工程规则：预设轮灌表 30 组 × 4 h（故障时去掉被隔离和已灌支管）；切换用规则档位 0（规范最短关阀时间）
消融（--ablation）：no_prompt（去掉硬约束与经验规则两段）、no_transient（反馈校核不做瞬态）、no_envelope（去掉 screen_transient 工具）
同一 (场景, 重复号) 下四类决策器看到的状态（切换的组对、故障发生时刻）相同；最终方案一律用全量 check_constraints 复核。

每条记录：maker, model, ablation, task, scenario, split, repeat, feasible_before_fallback, fallback_used, final_feasible,
  n_rounds, tool_calls, tokens{prompt, completion, cache_hit, cache_miss}, cost_cny（按调用时段）/cost_peak_cny/cost_offpeak_cny（prices.json），
  latency{model_s, wall_s, first_feasible_s},
  in_window, objectives, score(C*), gap_to_optimizer（C*_opt − C*），transient 极值, action_span_s, violations

python evalrun.py --maker rules optimizer --task rotation_plan switch fault_response --n 5
python evalrun.py --maker workflow prompt_only --task switch --n 5 --model deepseek-v4-pro --tag pre_switch
python evalrun.py --maker workflow --task switch rotation_plan fault_response --set control40 --shard 0/4 --tag main_kimi ...
正式实验的批次（场景集、并发路数、标签）在 batches.json，用 runbatch.py 启动。
"""
import argparse
import json
import pathlib
import time
import numpy as np
import network as net
import scoring
import tools
import teacher
import transient_tools as tt

HERE = pathlib.Path(__file__).parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
PRICES = json.load(open(HERE / "prices.json", encoding="utf-8")) if (HERE / "prices.json").exists() else {}
LLM_MAKERS = ("workflow", "prompt_only")
ABLATIONS = {"none": {}, "no_prompt": dict(ablate=True), "no_transient": dict(skip_transient=True),
             "no_envelope": dict(tool_specs=[s for s in tools.TOOL_SPECS if s["name"] != "screen_transient"]),
             # 三个组件同时去掉（难瞬态子集的可选对照：单项消融无效应时检验组件是否互为冗余）
             "no_all": dict(ablate=True, skip_transient=True, tool_specs=[s for s in tools.TOOL_SPECS if s["name"] != "screen_transient"])}


def safety_of(ctx, decision, state):
    """难瞬态子集的安全口径（--safety）：不论该配置做不做瞬态校核，都对方案做全量复核和全量瞬态，
    返回 full_feasible、未通过的检查项和 transient_tools.safety_summary（最大水锤压力、最坏比、越限点数）。"""
    if not decision:
        return None
    acts = (decision.get("switch") or {}).get("actions") or (decision.get("fault_response") or {}).get("isolation_actions") or []
    try:
        chk = tools.check_constraints(ctx, decision, open_before=state.get("g1"))
        s = dict(full_feasible=chk["feasible"], failed=[k for k, v in chk["checks"].items() if not v["feasible"]])
        if acts:
            s.update(tt.safety_summary(tt.simulate_switch(ctx, state.get("g1") or [], [tools.project_action(x) for x in acts])))
        return s
    except Exception as e:
        return dict(full_feasible=False, error=f"{type(e).__name__}: {str(e)[:200]}")


def load_scenarios():
    """主场景库 + 难瞬态子集（make_hard.py → scenarios_hard.jsonl，split = hard，不进 --split 抽样）。"""
    S = [json.loads(l) for l in open(HERE / "scenarios.jsonl", encoding="utf-8")]
    if (HERE / "scenarios_hard.jsonl").exists():
        S += [json.loads(l) for l in open(HERE / "scenarios_hard.jsonl", encoding="utf-8")]
    return S


def scenario_rng(seed, sc, repeat):
    return np.random.default_rng([seed, repeat, int(sc["id"][1:])])


def is_peak(t=None):
    """DeepSeek 高峰时段：北京时间工作日 9–12、14–18（不判法定节假日）。
    显式按 UTC+8 算，不用本机时区——本机时区是新西兰（UTC+13），比北京快 5 h，2026-10-28 门禁与 peak 字段因此错位过。"""
    lt = time.gmtime((time.time() if t is None else t) + 8 * 3600)
    return lt.tm_wday < 5 and (9 <= lt.tm_hour < 12 or 14 <= lt.tm_hour < 18)


def wait_offpeak(provider):
    """省钱门禁（作者 2026-10-28 11:4x 批准）：results/ds_offpeak_only.txt 为 1 时，DeepSeek 的新场景只在空闲时段发起；
    高峰时段在场景边界等待（在途场景照常跑完），等待不计入该行时延。返回等待秒数。"""
    if provider != "deepseek":
        return 0.0
    t0 = time.time()
    while is_peak():
        try:
            if (HERE / "results" / "ds_offpeak_only.txt").read_text().strip() != "1":
                break
        except OSError:
            break
        time.sleep(60)
    return round(time.time() - t0, 1)


def tokens_cost(usage, model, t=None):
    """返回 token 分项和费用（元）：按调用时段计价，并同时给出高峰、空闲两种价格下的费用。"""
    tok = dict(prompt=0, completion=0, cache_hit=0, cache_miss=0, cache_write=0, reasoning=0)
    for u in usage:
        tok["prompt"] += u.get("prompt_tokens") or 0
        tok["completion"] += u.get("completion_tokens") or 0
        pd_, cd_ = u.get("prompt_tokens_details") or {}, u.get("completion_tokens_details") or {}
        if "prompt_cache_hit_tokens" in u:                      # DeepSeek 字段
            hit = u.get("prompt_cache_hit_tokens") or 0
        else:                                                   # OpenAI 通用字段（Kimi 等）
            hit = pd_.get("cached_tokens") or 0
        tok["cache_hit"] += hit
        tok["cache_miss"] += (u.get("prompt_tokens") or 0) - hit
        tok["cache_write"] += pd_.get("cache_write_tokens") or 0   # Kimi K3 缓存写入另计费
        tok["reasoning"] += cd_.get("reasoning_tokens") or 0
    p = PRICES.get(model)
    if not p:
        return tok, dict(cost_cny=None, cost_peak_cny=None, cost_offpeak_cny=None, peak=is_peak(t))
    cost = {k: round((tok["cache_hit"] * v["input_cache_hit"] + tok["cache_miss"] * v["input_cache_miss"]
                      + tok["cache_write"] * v.get("cache_write", 0) + tok["completion"] * v["output"]) / 1e6, 5)
            for k, v in p.items() if isinstance(v, dict)}
    peak = is_peak(t)
    return tok, dict(cost_cny=cost["peak" if peak else "offpeak"], cost_peak_cny=cost["peak"], cost_offpeak_cny=cost["offpeak"], peak=peak)


def save_ref(path, key, val):
    """优化器参照的多进程安全写：重读文件合并后原子替换（分片并行时各进程只加自己的键，不互相覆盖）。"""
    import os
    for _ in range(5):
        try:
            cur = json.load(open(path, encoding="utf-8")) if path.exists() else {}
            break
        except (json.JSONDecodeError, OSError):
            time.sleep(0.2)
    else:
        cur = {}
    cur[key] = val
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    json.dump(cur, open(tmp, "w", encoding="utf-8"))
    for _ in range(10):
        try:
            os.replace(tmp, path); break
        except PermissionError:
            time.sleep(0.2)
    return cur


def rules_maker(task, sc, state, ctx):
    if task == "rotation_plan":
        plan = dict(groups=[dict(group_id=k + 1, valves=g, duration_h=4.0) for k, g in enumerate(net.rotation_groups())],
                    order=list(range(1, 31)))
        return dict(decision_type="rotation_plan", rotation_plan=plan), {}
    if task == "switch":
        acts = tools.rule_switch(state["g1"], state["g2"], 0)
        return dict(decision_type="switch", switch=dict(from_group=state["from_group"], to_group=state["to_group"], actions=acts)), {}
    return tools.rule_fault(ctx, sc), {}


def optimizer_maker(task, sc, state, ctx):
    if task == "switch":
        acts, r, lv = tools.safe_switch(ctx, state["g1"], state["g2"])
        return dict(decision_type="switch", switch=dict(from_group=state["from_group"], to_group=state["to_group"], actions=acts)), dict(level=lv)
    r = tools.run_optimizer(ctx, sc, apply_faults=(task == "fault_response"))
    if task == "rotation_plan":
        return dict(decision_type="rotation_plan", rotation_plan=r["rotation_plan"]), dict(cpu_s=r["cpu_s"])
    return tools.rule_fault(ctx, sc, r["rotation_plan"]), dict(cpu_s=r["cpu_s"])


def final_check(ctx, task, decision, state):
    """最终方案的全量复核（所有决策器一致）。"""
    try:
        chk = tools.check_constraints(ctx, decision, open_before=state.get("g1"))
    except Exception as e:
        return dict(final_feasible=False, check_error=f"{type(e).__name__}: {str(e)[:200]}")
    c = chk["checks"]
    rec = dict(final_feasible=chk["feasible"])
    if "steady" in c:
        rec.update(objectives=c["steady"]["objectives"], score=scoring.closeness(c["steady"]["objectives"]),
                   violations=c["steady"]["violations"])
    if "transient" in c:
        rec.update(transient=c["transient"]["extremes"], transient_violations=c["transient"]["n_violations"])
    if "tau_min" in c:
        rec["tau_violations"] = len(c["tau_min"]["violations"])
    acts = decision.get("switch", {}).get("actions") or decision.get("fault_response", {}).get("isolation_actions") or []
    if acts:
        rec["action_span_s"] = round(max(tt.action_end(a) for a in acts), 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maker", nargs="+", default=["rules"], choices=["workflow", "prompt_only", "optimizer", "rules"])
    ap.add_argument("--task", nargs="+", default=["switch"], choices=["rotation_plan", "switch", "fault_response"])
    ap.add_argument("--ablation", default="none", choices=list(ABLATIONS))
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--key-env", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-done", action="store_true", help="同一 tag 文件里已有的 (决策器, 模型, 消融, 任务, 场景, 重复) 不再跑，省 API 费用")
    ap.add_argument("--set", default=None, help="sets/<名>.json 的固定场景集（make_sets.py 生成），给了就不看 --split；--n 仍取前 n 个")
    ap.add_argument("--shard", default="0/1", help="i/k：只跑场景序号 mod k == i 的部分，写 <tag>.s<i>of<k>.jsonl，供单家多路并行")
    ap.add_argument("--outdir", default="results", help="结果目录（相对 src）；难瞬态子集用 results/hard，不混进主实验的 results/*.jsonl")
    ap.add_argument("--safety", action="store_true", help="难瞬态子集：每轮提交方案和最终方案都做全量安全口径（safety_of），记 rounds_safety / safety_final")
    a = ap.parse_args()

    S = load_scenarios()
    client = teacher.Client(a.model, a.provider, a.key_env, a.dry_run) if any(m in LLM_MAKERS for m in a.maker) else None
    OUT = HERE / a.outdir
    OUT.mkdir(parents=True, exist_ok=True)
    ref_path = OUT / "opt_ref.json"
    ref = json.load(open(ref_path, encoding="utf-8")) if ref_path.exists() else {}
    si, sk = map(int, a.shard.split("/"))
    out = OUT / (f"{a.tag}.jsonl" if sk == 1 else f"{a.tag}.s{si}of{sk}.jsonl")
    fixed = json.load(open(HERE / "sets" / f"{a.set}.json", encoding="utf-8")) if a.set else None
    by_id = {s["id"]: s for s in S}
    done = set()
    if a.skip_done:                                              # 同一 tag 的所有分片文件互认，改路数或校准先跑的都不重跑
        import glob
        for fn in glob.glob(str(OUT / f"{a.tag}.jsonl")) + glob.glob(str(OUT / f"{a.tag}.s*of*.jsonl")):
            for l in open(fn, encoding="utf-8"):
                r = json.loads(l)
                done.add((r["maker"], r.get("model"), r.get("ablation"), r["task"], r["scenario"], r["repeat"]))
    f = open(out, "a", encoding="utf-8")
    for task in a.task:
        kind = "fault" if task == "fault_response" else "nominal"
        scs = [by_id[i] for i in fixed[kind]][:a.n] if fixed else [s for s in S if s["split"] == a.split and s["kind"] == kind][:a.n]
        scs = scs[si::sk]
        for sc in scs:
            for rep in range(a.repeats):
                ctx, state = teacher.make_state(task, sc, scenario_rng(a.seed, sc, rep))
                key = f"{task}|{sc['id']}|{rep}"
                if ref_path.exists():                            # 重读，拿到其它分片 / 基线批次已算好的参照
                    try:
                        ref = json.load(open(ref_path, encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        pass
                for maker in a.maker:
                    rec = dict(maker=maker, model=a.model if maker in LLM_MAKERS else None, ablation=a.ablation if maker in LLM_MAKERS else None,
                               task=task, scenario=sc["id"], split=sc["split"], repeat=rep, tag=a.tag)
                    if (maker, rec["model"], rec["ablation"], task, sc["id"], rep) in done:
                        continue
                    offpeak_wait = wait_offpeak(a.provider) if maker in LLM_MAKERS else 0.0
                    t0 = time.time()
                    if maker in LLM_MAKERS:
                        flags = dict(ABLATIONS[a.ablation])
                        if maker == "prompt_only":
                            flags.update(k_rounds=0, tool_specs=[])
                        try:
                            log = teacher.solve(client, task, sc, scenario_rng(a.seed, sc, rep), **flags)
                        except Exception as e:                   # 超时 / 重试耗尽 / 其它 API 错误：记一条 status 行，跳到下一场景，不中断批次
                            status = "timeout" if "Timeout" in type(e).__name__ else "error"
                            rec.update(status=status, error=f"{type(e).__name__}: {str(e)[:300]}", feasible_before_fallback=None,
                                       fallback_used=None, n_rounds=0, tool_calls=None, tokens=None, cost_cny=None, cost_peak_cny=None,
                                       cost_offpeak_cny=None, peak=is_peak(), served_model=None, final_feasible=None, decision=None,
                                       latency=dict(model_s=None, wall_s=round(time.time() - t0, 1), first_feasible_s=None), in_window=False)
                            rec["t_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())   # 落盘时刻（UTC），事后可按北京时间重算分时费用
                            f.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n"); f.flush()
                            print(f"{task} {sc['id']} r{rep} {maker:<11} {status}: {rec['error']}", flush=True)
                            continue
                        decision = log["final_decision"]
                        tok, cost = tokens_cost(log["usage"], a.model)
                        rec.update(feasible_before_fallback=log["feasible_before_fallback"], fallback_used=log["fallback_used"],
                                   n_rounds=log["n_rounds"], tool_calls=log["tool_calls"], tokens=tok, **cost,
                                   latency=dict(model_s=round(log["latency_s"], 1), wall_s=round(log["wall_s"], 1),
                                                first_feasible_s=None if log["first_feasible_s"] is None else round(log["first_feasible_s"], 1),
                                                retry_wait_s=sum(u.get("retry_wait_s") or 0 for u in log["usage"])),
                                   in_window=log["in_window"], served_model=next((u.get("served_model") for u in log["usage"] if u.get("served_model")), None),
                                   offpeak_wait_s=offpeak_wait)
                        if a.safety:                             # 每轮提交方案的全量安全口径（含该配置没做的瞬态校核），模型自己的最后方案单独留存
                            rec.update(model_decision=log["decision"], fallback_level=log.get("fallback_level"),
                                       rounds_safety=[dict(round=r["round"], parsed=r["decision"] is not None, passed_own_check=r["check"]["feasible"],
                                                           own_check_failed=[k for k, v in r["check"].get("checks", {}).items() if not v["feasible"]]
                                                           or (["format"] if "format_error" in r["check"] else []),
                                                           **(safety_of(ctx, r["decision"], state) or {})) for r in log["rounds"]])
                    else:
                        try:
                            decision, info = (rules_maker if maker == "rules" else optimizer_maker)(task, sc, state, ctx)
                        except Exception as e:                   # 退化场景（如故障隔离掉全部阀门，GA 无可排，2026-11-01 O00237）：记 error 行，不中断批次
                            rec.update(status="error", error=f"{type(e).__name__}: {str(e)[:300]}", feasible_before_fallback=None, fallback_used=None,
                                       n_rounds=0, tool_calls=0, tokens=None, cost_cny=0.0, final_feasible=None, decision=None,
                                       latency=dict(model_s=0.0, wall_s=round(time.time() - t0, 1), first_feasible_s=None), in_window=False)
                            rec["t_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                            f.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n"); f.flush()
                            print(f"{task} {sc['id']} r{rep} {maker:<11} error: {rec['error']}", flush=True)
                            continue
                        wall = time.time() - t0
                        rec.update(feasible_before_fallback=None, fallback_used=None, n_rounds=0, tool_calls=0, tokens=None, cost_cny=0.0,
                                   latency=dict(model_s=0.0, wall_s=round(wall, 1), first_feasible_s=None),
                                   in_window=wall <= teacher.WINDOW_S[task], **info)
                    rec.update(final_check(ctx, task, decision, state))
                    rec["decision"] = decision
                    if a.safety:
                        rec["safety_final"] = safety_of(ctx, decision, state)
                    if maker == "optimizer" and rec.get("score") is not None:
                        ref = save_ref(ref_path, key, dict(score=rec["score"], feasible=rec["final_feasible"]))
                    if task != "switch" and rec.get("score") is not None:
                        if key not in ref:                       # 优化器参照：缺就算一次并缓存
                            try:
                                d, _ = optimizer_maker(task, sc, state, ctx)
                                rr = final_check(ctx, task, d, state)
                                ref = save_ref(ref_path, key, dict(score=rr.get("score"), feasible=rr["final_feasible"]))
                            except Exception as e:               # 参照算不出（退化场景）：本行不报差距、不写缓存，行照常落盘
                                rec["ref_error"] = f"{type(e).__name__}: {str(e)[:200]}"
                        if key in ref and ref[key]["score"] is not None:
                            rec["gap_to_optimizer"] = round(ref[key]["score"] - rec["score"], 4)
                    rec["t_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    f.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n"); f.flush()
                    print(f"{task} {sc['id']} r{rep} {maker:<11} 可行(兜底前) {rec.get('feasible_before_fallback')} 最终 {rec['final_feasible']} "
                          f"兜底 {rec.get('fallback_used')} {rec['latency']['wall_s']} s 窗口{'内' if rec['in_window'] else '外'} "
                          f"C* {rec.get('score')} 差距 {rec.get('gap_to_optimizer')} 费用 {rec.get('cost_cny')}", flush=True)
    f.close()
    print("→", out.relative_to(HERE))


if __name__ == "__main__":
    main()
