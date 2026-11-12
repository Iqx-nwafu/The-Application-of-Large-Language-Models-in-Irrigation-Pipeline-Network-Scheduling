"""难瞬态子集（hard50）：让瞬态约束成为真约束的 25 个切换场景 + 25 个爆管故障场景，检验三个工作流组件（约束提示、瞬态校核、安全包络）。

背景：主实验消融（Table 6）三个组件没有效应。查明两点原因：
  1. 切换任务：主实验场景里模型按提示的默认动作（开阀 10 s 线性、两阶段关 20 s、错开 2 s）全部安全，连直出都 0 越限；
  2. 故障任务：主实验口径里隔离阀在初始稳态已关闭（Context.closed_links），关隔离阀本身不产生水锤，瞬态校核实际只看到停当前组的动作。
所以难子集做两类：
  S（切换，H00000–H00024）：跨分干管切换（Jx5+Jx6 或 Jx3+Jx4 → Jy1+Jy2，x ≠ y），叠加高水头（30/35 m，perturbation.h_source）
     和高波速（×1.0–1.5，perturbation.wavespeed_mult），属于参数应力测试，不是分布内抽样。
  F（故障，H01000–H01024）：分布内爆管场景（分干管爆管关 VSx、运行中支管爆管关支管阀），物理参数不变，
     只把隔离关阀放进瞬态校核（sc["iso_transient"]：从爆管出流中、隔离阀开着的稳态起算，隔离段关阀后不作约束）。
入选条件（全部用 src 的同一套复核代码在 MOC 内核上算）：
  基线超限 —— 无组件的动作会越限：S 用"去提示风格"（开/关 30 s 线性、同时动作，取自 ds_no_prompt 实际方案的中位数），
              F 用工程规则（规范最短历时线性关，tools.rule_fault level 0）；
  有解     —— 兜底逐档放慢能找到可行动作（S：tools.safe_switch；F：tools.safe_fault）；
  非退化   —— 故障后仍有可调度支管（剩余支管 ≥ 4，预设表评价不报错）；分干管 J7（分布外留出位置）不进。
  难度分档：A 档 = 提示默认动作也越限（必须靠瞬态工具或反馈改慢）；B 档 = 只有无提示风格越限。按 A 档优先、位置与参数分散选取。

python make_hard.py    → scenarios_hard.jsonl、sets/hard50.json、sets/hard50_probe.csv（每个候选在各风格下的越限点数与最坏压力比）
"""
import os
import sys
import json
import pathlib
import collections
for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(k, "1")
import numpy as np
from multiprocessing import Pool
import network as net

HERE = pathlib.Path(__file__).parent
N_EACH = 25
HOLDOUT = 7
PHYS = [(1.0, 35.0), (1.25, 30.0), (1.25, 35.0), (1.5, 30.0), (1.5, 35.0)]      # (波速倍数, 水源水头 m)
STYLE = {   # 切换动作风格：(开阀模板, 关阀模板, 相邻错开 s, 先开后关间隔 s)
    "prompt_default": (("open_linear", dict(duration_s=10)), ("close_two_stage", dict(duration_s=20, break_time_s=5, break_opening=0.3)), 2, 4),
    "no_prompt": (("open_linear", dict(duration_s=30)), ("close_linear", dict(duration_s=30)), 0, 0),
    "fast": (("open_linear", dict(duration_s=5)), ("close_linear", dict(duration_s=5)), 0, 0),
}
# 故障隔离风格：提示默认（主实验模型常用：分干管阀 close_linear 10 s，支管阀 close_two_stage 20/5/0.3）与 instant
ISO_STYLE = {"prompt_default": {"sub": ("close_linear", dict(duration_s=10)), "branch": ("close_two_stage", dict(duration_s=20, break_time_s=5, break_opening=0.3))},
             "instant": {"sub": ("instant", {}), "branch": ("instant", {})}}


def style_actions(name, g1, g2):
    (ot, op), (ct, cp), gap, lag = STYLE[name]
    acts, t = [], 0.0
    for v in sorted(g2):
        acts.append(dict(valve=v, op="open", t_start_s=t, template=ot, params=dict(op))); t += gap
    t = acts[-1]["t_start_s"] + lag
    for v in sorted(g1):
        acts.append(dict(valve=v, op="close", t_start_s=t, template=ct, params=dict(cp))); t += gap
    return acts


def _summ(ctx, open_before, acts):
    import tools, transient_tools as tt
    out = tt.simulate_switch(ctx, open_before, [tools.project_action(a) for a in acts])
    s = tt.safety_summary(out)
    return dict(n_viol=s["n_viol"], worst_ratio=s["worst_ratio"], p_max_pvc=s["p_max_pvc"], p_max_pe=s["p_max_pe"], p_min=s["p_min"])


def _clear():
    """每个候选的糙率不同，各建一份内核；清缓存免得进程内存越积越多。"""
    import transient_tools as tt, evaluate as ev
    tt._sims.clear(); ev._models.clear()


def eval_switch(sc):
    import teacher, tools
    _clear()
    ctx, st = teacher.make_state("switch", sc, np.random.default_rng(0))
    r = dict(id=sc["id"], family="S", g1=st["from_group"], g2=st["to_group"], w=sc["perturbation"]["wavespeed_mult"],
             h=sc["perturbation"]["h_source"])
    for k in STYLE:
        r.update({f"{k}_{x}": v for x, v in _summ(ctx, st["g1"], style_actions(k, st["g1"], st["g2"])).items()})
    acts, rr, lv = tools.safe_switch(ctx, st["g1"], st["g2"])
    r.update(safe_level=lv, solvable=bool(rr["feasible"]))
    r["baseline_viol"] = r["no_prompt_n_viol"] > 0
    r["tier"] = "A" if r["prompt_default_n_viol"] > 0 else "B"
    return r


def eval_fault(sc):
    import teacher, tools, evaluate as ev
    _clear()
    ctx, st = teacher.make_state("fault_response", sc, np.random.default_rng(0))
    f = sc["faults"][0]
    r = dict(id=sc["id"], family="F", source=sc["source"], pipe=f["pipe"], pipe_class=f["pipe_class"], iso=f["isolate_valves"][0],
             running=st["g1"][0][:3], n_left=len(ctx.valves), t_left_h=round(st["t_left_h"], 1))
    groups = [g for g in ([v for v in g if v in ctx.valves] for g in net.rotation_groups()) if g]
    try:
        ev.evaluate(ctx, groups)
        r["degenerate"] = len(ctx.valves) < 4
    except Exception:
        r["degenerate"] = True
    for lv in range(4):
        d = tools.rule_fault(ctx, sc, None, lv)
        r.update({f"rule{lv}_{x}": v for x, v in _summ(ctx, st["g1"], d["fault_response"]["isolation_actions"]).items()})
    for k, spec in ISO_STYLE.items():
        acts = [dict(valve=v, op="close", t_start_s=0.0, template=spec["sub" if v.startswith("VS") else "branch"][0],
                     params=dict(spec["sub" if v.startswith("VS") else "branch"][1])) for v in f["isolate_valves"]]
        r.update({f"{k}_{x}": v for x, v in _summ(ctx, st["g1"], acts).items()})
    ok = [lv for lv in range(4) if r[f"rule{lv}_n_viol"] == 0]
    r.update(safe_level=ok[0] if ok else None, solvable=bool(ok), baseline_viol=r["rule0_n_viol"] > 0,
             tier="A" if r["prompt_default_n_viol"] > 0 else "B")
    return r


def candidates():
    S = [json.loads(l) for l in open(HERE / "scenarios.jsonl", encoding="utf-8")]
    G = net.rotation_groups()
    # 切换：源场景取分布内标称（需水、糙率），切换组对与物理参数按网格枚举
    nom = [s for s in S if s["kind"] == "nominal" and s["split"] in ("val", "test")]
    pairs = [(3 * x + t, 3 * y) for x in range(10) for y in range(10) for t in (1, 2) if x != y and HOLDOUT not in (x, y)]
    sw, i = [], 0
    for (k1, k2) in pairs:
        for w, h in PHYS:
            src = nom[i % len(nom)]; i += 1
            pert = dict(src["perturbation"], wavespeed_mult=w, h_source=h, stress_test=True)
            sw.append(dict(id=f"C{len(sw):05d}", kind="nominal", irrigation_event=src["irrigation_event"], quota_m3_per_mu=src["quota_m3_per_mu"],
                           window=src["window"], perturbation=pert, demand_m3=src["demand_m3"], switch_pair=[k1, k2], source=src["id"]))
    # 故障：分布内爆管（分干管、支管），物理参数不变，隔离关阀计入瞬态；支管爆管把发生时刻放进该支管所在组的运行时段
    fl = []
    for s in S:
        if s["kind"] != "fault" or s["split"] == "ood" or len(s["faults"]) != 1 or s["faults"][0]["type"] != "burst":
            continue
        f = dict(s["faults"][0])
        if f["submain"] == HOLDOUT:
            continue
        if f["pipe_class"] == "branch":
            k = next(j for j, g in enumerate(G) if f["isolate_valves"][0] in g)
            f.update(phase="running", cycle_frac=round((k + 0.5) / len(G), 4)); f.pop("switch_frac", None); f.pop("t_in_switch_s", None)
        fl.append(dict({x: s[x] for x in ("irrigation_event", "quota_m3_per_mu", "window", "perturbation", "demand_m3")},
                       id=f"D{len(fl):05d}", kind="fault", faults=[f], iso_transient=True, source=s["id"]))
    return sw, fl


MIN_EXCESS = 1.03          # A 档要求提示默认动作的最坏压力比 ≥ 1.03（越限 ≥ 3%），免得零点几米的边缘越限被模型的小改动带过


def greedy(rows, n, key, caps, uniq=None):
    """按 key 从大到小贪心取 n 个；caps = [(分组函数, 每组上限)] 保证位置 / 参数分散；uniq 为去重键（同一管段只取一次）。"""
    out, cnt, seen = [], collections.Counter(), set()
    for r in sorted(rows, key=key, reverse=True):
        if len(out) >= n:
            break
        u = uniq(r) if uniq else r["id"]
        if u in seen or any(cnt[(i, f(r))] >= c for i, (f, c) in enumerate(caps)):
            continue
        out.append(r); seen.add(u)
        for i, (f, c) in enumerate(caps):
            cnt[(i, f(r))] += 1
    return out


def select(rs, rf):
    """切换：A 档且越限 ≥ 3%，切换组对不重复，每档物理参数 ≤ 7 个、每个起始分干管 ≤ 10 个（难切换集中在远端 8、9 号分干管切回近端，物理上如此）。
    故障：分干管爆管 13 个（A 档，管段不重复）；运行中支管爆管 12 个（管段不重复，先每条分干管 1 个，再补到每条 ≤ 2 个）。"""
    xs = lambda r: (int(r["g1"]) - 1) // 3
    ok_s = [r for r in rs if r["baseline_viol"] and r["solvable"] and r["tier"] == "A" and r["prompt_default_worst_ratio"] >= MIN_EXCESS]
    sel_s = greedy(ok_s, N_EACH, lambda r: r["prompt_default_worst_ratio"],
                   caps=[(lambda r: (r["w"], r["h"]), 7), (xs, 10)], uniq=lambda r: (int(r["g1"]), int(r["g2"])))
    ok_f = [r for r in rf if r["baseline_viol"] and r["solvable"] and not r["degenerate"]]
    sub = greedy([r for r in ok_f if r["pipe_class"] == "sub"], 13, lambda r: r["rule0_worst_ratio"], caps=[(lambda r: r["iso"], 4)], uniq=lambda r: r["pipe"])
    brs = [r for r in ok_f if r["pipe_class"] == "branch"]
    kb = lambda r: (r["tier"] == "A", r["rule0_worst_ratio"])
    br = greedy(brs, 12, kb, caps=[(lambda r: r["iso"][1], 1)], uniq=lambda r: r["pipe"])
    used = {r["pipe"] for r in br}
    br += greedy([r for r in brs if r["pipe"] not in used], 12 - len(br), kb,        # 第二轮每条分干管再补 1 个
                 caps=[(lambda r: r["iso"][1], 1)], uniq=lambda r: r["pipe"])
    return sel_s, sub + br, ok_s, ok_f


if __name__ == "__main__":
    import pandas as pd
    sw, fl = candidates()
    print(f"候选：切换 {len(sw)}，爆管故障 {len(fl)}")
    if "--select-only" in sys.argv:                       # 只按已有 hard50_probe.csv 重新选，不重算内核
        probe = pd.read_csv(HERE / "sets" / "hard50_probe.csv")
        recs = [{k: v for k, v in r.items() if not (isinstance(v, float) and np.isnan(v))} for r in probe.to_dict("records")]
        rs, rf = [r for r in recs if r["family"] == "S"], [r for r in recs if r["family"] == "F"]
    else:
        with Pool(int(os.environ.get("HARD_WORKERS", 10))) as p:
            rs = p.map(eval_switch, sw, chunksize=4)
            rf = p.map(eval_fault, fl, chunksize=4)
        probe = pd.DataFrame(rs + rf)
        probe.to_csv(HERE / "sets" / "hard50_probe.csv", index=False, encoding="utf-8-sig")
    sel_s, sel_f, ok_s, ok_f = select(rs, rf)
    print("切换：基线超限且有解", sum(bool(r["baseline_viol"]) and bool(r["solvable"]) for r in rs), "，其中 A 档且越限 ≥ 3%", len(ok_s),
          "；故障：基线超限、有解、非退化", len(ok_f), "（A 档", sum(r["tier"] == "A" for r in ok_f), "，退化剔除", sum(bool(r["degenerate"]) for r in rf), "）")
    by = {s["id"]: s for s in sw + fl}
    out, ids = [], dict(nominal=[], fault=[])
    for j, r in enumerate(sorted(sel_s, key=lambda r: r["id"])):
        sc = dict(by[r["id"]], id=f"H{j:05d}", split="hard", hard=dict(family="S", tier=r["tier"], candidate=r["id"], safe_level=r["safe_level"]))
        out.append(sc); ids["nominal"].append(sc["id"])
    for j, r in enumerate(sorted(sel_f, key=lambda r: r["id"])):
        sc = dict(by[r["id"]], id=f"H{1000 + j:05d}", split="hard", hard=dict(family="F", tier=r["tier"], candidate=r["id"], safe_level=r["safe_level"]))
        out.append(sc); ids["fault"].append(sc["id"])
    with open(HERE / "scenarios_hard.jsonl", "w", encoding="utf-8") as f:
        for sc in out:
            f.write(json.dumps(sc, ensure_ascii=False) + "\n")
    json.dump(dict(ids, _note="难瞬态子集（make_hard.py）：nominal 只跑 switch，fault 跑 fault_response；候选与入选依据见 hard50_probe.csv"),
              open(HERE / "sets" / "hard50.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sel = pd.DataFrame(sel_s + sel_f)
    print(sel.groupby(["family", "tier"]).size())
    cols = ["id", "family", "tier", "safe_level", "w", "h", "g1", "g2", "pipe", "iso", "no_prompt_n_viol", "prompt_default_n_viol",
            "rule0_n_viol", "no_prompt_worst_ratio", "prompt_default_worst_ratio", "rule0_worst_ratio"]
    print(sel[[c for c in cols if c in sel]].to_string())
