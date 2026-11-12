"""读 evalrun 结果为整洁表（只读 results/，不写）。

一行 = 一次决策（实体 × 任务 × 场景 × 重复 × 批次 tag）。实体 id 见 style.ENTITIES。
费用按 prices.json 和 token 分项重算（offpeak / peak 两列，非 DeepSeek 两列相同），改 usd_cny 后重跑即可。
超时（status=timeout）分两种并单独标注：timeout_kind = "not_attempted"（千问 Max 分组，按指示未发起）/ "timeout"（真实超时）。
处理口径由 CONFIG["timeout"] 决定（run_all --timeout）：
  exclude（默认，同 stats.py）：只计入 n、超时计数和实时达标率（记未达标），可行率、C*、费用、时延一律剔除；
  fail：记不可行、未达标、C* = 0（无方案 = 最差贴近度），费用与时延仍剔除。
status=error（账单欠费、重试耗尽等非模型原因）整行剔除，不计入超时、不进任何分母；条数在 load 时打印，补跑后的 ok 行优先。
tool_cap_hit：每轮 12 次工具调用上限是否命中；结果行有该字段就用，没有时用 tool_calls ≥ 13 作代理
（蒸馏试点校准：召回 28/28、精确 28/29）。
隔离与计划一致性（iso_conflict，只读重判 decision，不重跑）：故障决策声明的隔离（isolate_valves，或 isolation_actions 里最后一次
操作为 close 的分干管阀 VSx）及其 removed_valves 覆盖的支管，若又出现在 remaining_plan 里即记不一致；关 VSx 覆盖该分干管全部
12 个支管。isolation_actions 里单关支管阀不算隔离（多为停当前组）。最后一次操作为 open 的阀视为已复开。
处理口径由 CONFIG["iso"] 决定（run_all --iso）：fail（默认，作者 2026-10-29 定正文用：计入不可行，兜底前可行记 False；
未用兜底的行，安全兜底层本应接管，最终可行与 C* 取同场景 NSGA-II 基线的结果——兜底与该基线是同一段代码 rule_fault + run_optimizer）
/ flag（附录：只标注，可行率与 C* 照原复核）。
根因：final_check 按场景真值建边界并要求未移除的支管全部入组（valve_unassigned），泄漏场景不移除任何支管，所以主动隔离泄漏段的方案
只有把被隔离支管照排进计划才能过复核（DeepSeek 推理里自述"为满足校核器对未分配支管阀的要求"）。
跨模型比较的场景集由 CONFIG["control"] 决定（run_all --control），默认 control60；有模型没跑全时交集自动缩小，
图表标 Provisional（千问两档 2026-10-28 补跑到 control60 中）。
iso_miss（只标注，不进任何口径）：爆管场景里方案关闭的阀没覆盖真实隔离阀（如把 P04 读成 4 号分干管、关 VS4），爆管实际未隔离。
"""
import glob
import json
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
STAGE = ROOT / "src"
RES = ROOT / "results" / "api"
MODEL_ALIAS = {"deepseek-chat": "deepseek-flash"}
LLM = {"workflow", "prompt_only"}
# twin_tol：时间窗判定容差（h），论文口径 0.03（同 src/evaluate.py 的 T_WIN_TOL）；None = 运行时的原判（1e-6）。
CONFIG = dict(timeout="exclude", control="control60", iso="fail", twin_tol=0.03)
TOOL_CAP = 12
SIDES = ("L", "R")
# 退化场景：所有批次整行剔除（作者 2026-11-01 定"从正文统计中剔除，在附录说明原因"）。
# O00237：故障关 VS9 后该轮灌事件 12 个支管阀全部移除，无可调度对象，评估器对空方案报错（优化器参照算不出）。
EXCLUDE_SCENARIOS = {"O00237"}
_FAULTS = {}


def faults():
    """场景 id → 故障列表（scenarios.jsonl，只读一次）。"""
    if not _FAULTS:
        for l in open(STAGE / "scenarios.jsonl", encoding="utf-8"):
            s = json.loads(l)
            if s.get("faults"):
                _FAULTS[s["id"]] = s["faults"]
    return _FAULTS


def fault_class(f):
    """leak / burst / stuck_open / stuck_closed / stuck_partial（双故障场景两项同类，取并集）。"""
    return "+".join(sorted({x["mode"] if x["type"] == "stuck" else x["type"] for x in f}))


def _closed(fr):
    """(声明隔离 + 关 VSx 动作 − 复开, 全部关阀动作 + 声明隔离 − 复开)。"""
    last = {}
    for a in sorted(fr.get("isolation_actions") or [], key=lambda a: a.get("t_start_s") or 0):
        last[a.get("valve")] = a.get("op")
    reopen = {v for v, o in last.items() if o == "open"}
    decl = set(fr.get("isolate_valves") or [])
    return (decl | {v for v, o in last.items() if o == "close" and str(v).startswith("VS")}) - reopen, \
           (decl | {v for v, o in last.items() if o == "close"}) - reopen


def iso_miss(fr, f):
    """爆管场景：方案关闭的阀没有覆盖真实隔离阀（关错阀，爆管实际未隔离）。评测器按场景真值隔离，测不出这类错误。"""
    if not all(x["type"] == "burst" for x in f):
        return None
    return not {v for x in f for v in x["isolate_valves"]} <= _closed(fr)[1]


def iso_check(fr):
    """fr = decision["fault_response"]。返回 (是否不一致, 类型 submain/branch, 计划里被隔离的支管数)。"""
    iso = _closed(fr)[0]
    vs = {v[2:] for v in iso if v.startswith("VS")}
    plan = {v for g in (fr.get("remaining_plan") or {}).get("groups", []) for v in g.get("valves", [])}
    sub = {v for v in plan if not v.startswith("VS") and v[1:2] in vs}
    br = plan & ({v for v in iso if not v.startswith("VS")} | set(fr.get("removed_valves") or []))
    kind = "+".join(k for k, x in (("submain", sub), ("branch", br - sub)) if x)
    return bool(sub | br), kind, len(sub | br)


def apply_iso_fail(df):
    """把不一致的 LLM 决策按"计入不可行"重判（见模块说明）。返回新表，原表不动。"""
    df = df.copy()
    for c in ("feasible_before", "final_feasible", "score"):           # 原复核值留底，Table 8 并排两种口径用
        if f"{c}_flag" not in df:
            df[f"{c}_flag"] = df[c]
    hit = (df.iso_conflict.fillna(False).astype(bool) & df.maker.isin(LLM)).values
    if not hit.any():
        return df
    ga = main_rows(df[(df.entity == "nsga2") & (df.task == "fault_response")]).set_index("scenario")
    fb = ~df.fallback.astype(bool).values
    sub = hit & fb                                     # 兜底前可行、本应被兜底接管的行
    sc = df.scenario.values[sub]
    df.loc[hit, "feasible_before"] = False
    df.loc[sub, "final_feasible"] = pd.array([ga.final_feasible.get(s, False) for s in sc], dtype="boolean")
    df.loc[sub, "score"] = [ga.score.get(s, np.nan) for s in sc]
    df.loc[sub, "fallback"] = True
    return df


def timeout_note():
    if CONFIG["timeout"] == "fail":
        return ("Time-outs (including requests not attempted) count as infeasible and late and are scored C* = 0; "
                "their cost and latency are unknown and excluded.")
    return ("Time-outs (including requests not attempted) count as late in the real-time rate and are excluded from "
            "feasibility, C*, cost and latency, as in the main-run statistics.")


def thresholds():
    return json.load(open(STAGE / "thresholds.json", encoding="utf-8"))


def prices():
    return json.load(open(STAGE / "prices.json", encoding="utf-8"))


def scenario_set(name):
    d = json.load(open(STAGE / "sets" / f"{name}.json", encoding="utf-8"))
    return {"nominal": list(d["nominal"]), "fault": list(d["fault"]), "all": list(d["nominal"]) + list(d["fault"])}


def batch_tags():
    """正式批次的 tag（来自 batches.json），预实验 / smoke 不读。"""
    b = json.load(open(STAGE / "batches.json", encoding="utf-8"))["batches"]
    return {x["tag"]: x for x in b}


def entity(maker, model, ablation):
    if maker == "optimizer":
        return "nsga2"
    if maker == "rules":
        return "rules"
    model = MODEL_ALIAS.get(model, model)
    if maker == "prompt_only":
        return "ds-direct" if model == "deepseek-v4-pro" else f"{model}-direct"
    if ablation and ablation != "none":
        return f"ds-{ablation}" if model == "deepseek-v4-pro" else f"{model}-{ablation}"
    return model


def _cost(tok, model, P):
    p = P.get(MODEL_ALIAS.get(model, model))
    if not tok or not p:
        return np.nan, np.nan
    out = []
    for k in ("offpeak", "peak"):
        v = p[k]
        out.append((tok.get("cache_hit", 0) * v["input_cache_hit"] + tok.get("cache_miss", 0) * v["input_cache_miss"]
                    + tok.get("cache_write", 0) * v.get("cache_write", 0) + tok.get("completion", 0) * v["output"]) / 1e6)
    return tuple(out)


def load(tags=None, results_dir=RES):
    """读全部正式批次；tags=None 时按 batches.json 的 tag 过滤。同一键重复出现（续跑重写）保留最后一条。"""
    tags = set(tags or batch_tags())
    P = prices()
    rows, errors = [], {}
    for fn in sorted(glob.glob(str(pathlib.Path(results_dir) / "*.jsonl"))):
        stem = pathlib.Path(fn).name.split(".")[0]
        if stem not in tags:
            continue
        for l in open(fn, encoding="utf-8"):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:                     # 批次正在写的半行
                continue
            if r.get("scenario") in EXCLUDE_SCENARIOS:
                continue
            lat, tok, tr, obj = r.get("latency") or {}, r.get("tokens") or {}, r.get("transient") or {}, r.get("objectives") or {}
            status = r.get("status", "ok")
            if status == "error":                            # 非模型原因的失败：剔除，补跑行会替代它
                errors[r["tag"]] = errors.get(r["tag"], 0) + 1
                continue
            to = status != "ok"
            fail = to and CONFIG["timeout"] == "fail"
            off, peak = _cost(tok, r.get("model"), P)
            wall = lat.get("wall_s")
            fr = (r.get("decision") or {}).get("fault_response") if r["task"] == "fault_response" else None
            iso = iso_check(fr) if fr else (None, "", None)
            rows.append(dict(
                tag=r["tag"], entity=entity(r["maker"], r.get("model"), r.get("ablation")), maker=r["maker"],
                model=MODEL_ALIAS.get(r.get("model"), r.get("model")), task=r["task"], scenario=r["scenario"],
                kind="fault" if r["task"] == "fault_response" else "nominal", repeat=r.get("repeat", 0),
                status=status, timeout_kind=("not_attempted" if r.get("not_attempted") else "timeout") if to else "",
                feasible_before=False if fail else (None if to else r.get("feasible_before_fallback")),
                final_feasible=False if fail else (None if to else r.get("final_feasible")),
                fallback=r.get("fallback_used") is not None,
                in_window=False if to else bool(r.get("in_window")),
                wall_s=np.nan if (to or wall is None) else wall - (lat.get("retry_wait_s") or 0),
                model_s=lat.get("model_s"), cpu_s=r.get("cpu_s"),
                cost_offpeak=off, cost_peak=peak, peak_call=r.get("peak"),
                tok_prompt=tok.get("prompt"), tok_completion=tok.get("completion"), tok_hit=tok.get("cache_hit"), tok_write=tok.get("cache_write", 0),
                tool_calls=r.get("tool_calls"), n_rounds=r.get("n_rounds"),
                tool_cap_hit=(bool(r["tool_cap_hit"]) if "tool_cap_hit" in r else
                              None if r.get("tool_calls") is None or r["maker"] not in LLM else r["tool_calls"] > TOOL_CAP),
                score=0.0 if fail else (np.nan if to else r.get("score")), gap_opt=r.get("gap_to_optimizer"),
                deficit=obj.get("deficit_m3"), cv_q=obj.get("cv_q"), T_total=obj.get("T_total_h"), over=obj.get("over_m3"),
                span=r.get("action_span_s"), p_max_ratio=tr.get("p_max_ratio"), p_min=tr.get("p_min"),
                p_min_branch=tr.get("p_min_branch"), transient_viol=r.get("transient_violations"),
                fault_type=fault_class(faults()[r["scenario"]]) if r["task"] == "fault_response" else "",
                iso_conflict=iso[0], iso_kind=iso[1], iso_n=iso[2],
                iso_miss=iso_miss(fr, faults()[r["scenario"]]) if fr else None,
                _viol=r.get("violations"), _tviol=r.get("transient_violations"), _tau=r.get("tau_violations"),
                _cerr=r.get("check_error"),
            ))
    if errors:
        print("[load] 剔除 status=error 行:", errors)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = (df.assign(_ok=(df.status == "ok").astype(int)).sort_values("_ok", kind="stable")   # 同键重复时 ok 行优先
          .drop_duplicates(["tag", "entity", "task", "scenario", "repeat"], keep="last").drop(columns="_ok")
          .sort_index().reset_index(drop=True))
    for c in ("feasible_before", "final_feasible", "tool_cap_hit", "iso_conflict", "iso_miss"):
        df[c] = df[c].astype("boolean")
    df = apply_twin_tol(df)
    return apply_iso_fail(df) if CONFIG["iso"] == "fail" else df


def apply_twin_tol(df):
    """时间窗容差重判（论文口径）：最终复核不可行、且唯一越限项是 T_window ≤ 容差（无瞬态 / τ 越限、无复核报错）的行改判可行。
    只用结果行里存的 violations，不重跑；在严格一致性口径（apply_iso_fail 取同场景 NSGA-II 结果）之前做，所有方法一致。"""
    tol = CONFIG.get("twin_tol")
    df["twin_flip"] = False
    if tol:
        def empty(x):                                   # 缺字段的行在表里是 NaN
            return x is None or (isinstance(x, float) and np.isnan(x)) or not x

        def flip(ff, v, tv, tau, ce):
            if ff is pd.NA or ff is None or bool(ff) or not isinstance(v, dict):
                return False
            return set(v) == {"T_window"} and v["T_window"] <= tol and empty(tv) and empty(tau) and empty(ce)
        df["twin_flip"] = [flip(*x) for x in zip(df.final_feasible, df._viol, df._tviol, df._tau, df._cerr)]
        df.loc[df.twin_flip, "final_feasible"] = True
    return df.drop(columns=["_viol", "_tviol", "_tau", "_cerr"])


def main_rows(df):
    """单次运行口径：去掉 ds_repro（30×10 重复）批次，同一实体 × 任务 × 场景只留一条（优先主批次 tag）。"""
    d = df[df.tag != "repro_ds"]
    order = {"main_ds": 0, "main_base": 0, "main_flash": 0}
    d = d.assign(_o=d.tag.map(order).fillna(1)).sort_values("_o")
    return d.drop_duplicates(["entity", "task", "scenario"], keep="first").drop(columns="_o")


def paired(df, entities, task, metric, scenarios=None, complete=True):
    """宽表：行 = 场景，列 = 实体；complete=True 只保留所有实体都有值的场景（配对）。"""
    d = main_rows(df)
    d = d[(d.task == task) & d.entity.isin(entities)]
    if scenarios is not None:
        d = d[d.scenario.isin(scenarios)]
    w = d.pivot_table(index="scenario", columns="entity", values=metric, aggfunc="first")
    w = w[[e for e in entities if e in w.columns]]
    return w.dropna() if complete else w


def common_scenarios(df, entities, tasks=("switch", "rotation_plan", "fault_response"), set_name=None):
    """对照集（默认 CONFIG["control"]）内、组内所有实体都有记录（含超时行）的场景，按任务分别给出。"""
    d = main_rows(df)
    ctl = scenario_set(set_name or CONFIG["control"])
    out = {}
    for t in tasks:
        s = set(ctl["fault"] if t == "fault_response" else ctl["nominal"])
        for e in entities:
            x = set(d[(d.entity == e) & (d.task == t)].scenario)
            if not x:
                continue                                    # 该实体尚无数据：不参与求交
            if s - x:
                print(f"[common_scenarios] {e} 缺 {len(s - x)} 个 {t} 场景（{CONFIG['control'] if set_name is None else set_name}），交集随之缩小")
            s = s & x
        out[t] = sorted(s)
    return out


def control_incomplete(scen, set_name=None):
    """共同场景比对照集少（有模型没跑全）时返回 True，图表据此标 Provisional。"""
    ctl = scenario_set(set_name or CONFIG["control"])
    return any(len(v) < len(ctl["fault"] if t == "fault_response" else ctl["nominal"]) for t, v in scen.items())


def present(df, entities):
    return [e for e in entities if e in set(df.entity)]


def progress(df):
    """各批次完成度（行数 / 计划行数），给图表打 provisional 标记用。"""
    B = batch_tags()
    out = []
    for tag, b in B.items():
        try:
            s = scenario_set(b["set"])
        except FileNotFoundError:
            continue
        tasks = b.get("tasks") or ["switch", "rotation_plan", "fault_response"]
        n_plan = sum(len(set(s["fault"] if t == "fault_response" else s["nominal"]) - EXCLUDE_SCENARIOS) for t in tasks)
        n_plan *= len(b["maker"]) * b.get("repeats", 1)
        n_done = int((df.tag == tag).sum()) if len(df) else 0
        out.append(dict(tag=tag, batch=b["name"], set=b["set"], done=n_done, planned=n_plan))
    p = pd.DataFrame(out).groupby("tag").agg(done=("done", "first"), planned=("planned", "max"),
                                             batches=("batch", lambda x: "+".join(x)), set=("set", "last")).reset_index()
    p["complete"] = p.done >= p.planned
    return p
