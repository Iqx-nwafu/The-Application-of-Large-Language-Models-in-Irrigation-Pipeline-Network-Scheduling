"""A4 需水情景、参数扰动和故障场景采样器。

需水：实施方案表 3.2-1（棉花 12 次灌水，第 1 次按 25.7 而非 30），D_i = A_i × m_t，A_i = 860 亩 / 120 条（无站工图时平均）。
扰动：糙率、波速、需水逐元素乘 (1 + U(-δ, δ))，δ ∈ {0.10, 0.15}；δ = 0.20 只进分布外组。
故障：爆管（分干管段、支管段；干管无分段阀，爆管等于全系统停水，不进采样器）、
      泄漏（干管、分干管、支管段）、阀门卡死（卡开、卡关、部分开度）。
位置按管长加权；分干管 J7 的全部管段和阀门为留出位置，只进分布外组。
时刻：70% 在某组运行期间（cycle_frac 均匀），30% 在组间切换窗口内（t_in_switch_s 均匀 0–60 s）。
划分：分布内按 70/15/15 分训练、验证、测试；分布外 = 留出位置 ∪ 留出需水（第 9 次灌水 22 m³/亩）
      ∪ δ = 0.20 ∪ 组合故障。

python scenarios.py  → scenarios.jsonl
"""
import json
import pathlib
import numpy as np
import network as net

HERE = pathlib.Path(__file__).parent
SYSTEM_AREA_MU = 860.0
N_BRANCH = 120

# 表 3.2-1：(次序, 灌水定额 m3/亩, 起, 止)
IRRIGATION_EVENTS = [
    # 第 1 次灌水表 3.2-1 为 30；作者 2026-10-27：按其算的田间需水 25.7 m³/亩（设计高峰定额）
    (1, 25.7, "04-10", "04-30"), (2, 23, "06-01", "06-10"), (3, 23, "06-11", "06-20"),
    (4, 25, "06-21", "06-27"), (5, 25, "06-28", "07-04"), (6, 25, "07-05", "07-11"),
    (7, 25, "07-12", "07-18"), (8, 25, "07-19", "07-25"), (9, 22, "07-26", "08-02"),
    (10, 20, "08-03", "08-10"), (11, 20, "08-11", "08-20"), (12, 20, "08-21", "08-31"),
]
HOLDOUT_EVENT = 9
HOLDOUT_SUBMAIN = 7
DELTAS_IN, DELTA_OOD = (0.10, 0.15), 0.20

# 爆管开口面积占管道断面的比例（按管径分档），泄漏流量范围
BURST_AREA_FRAC = {"sub": (0.10, 0.30), "branch": (0.20, 0.50)}
LEAK_Q_M3H = (1.0, 6.0)
CD = 0.6


def branch_area(rng=None, delta=0.0):
    """支管控制面积（亩）。平均分配，需水扰动作用在面积上。"""
    a = np.full(N_BRANCH, SYSTEM_AREA_MU / N_BRANCH)
    return a if rng is None else a * (1 + rng.uniform(-delta, delta, N_BRANCH))


def field_demand(quota, area):
    """D_i = A_i × m_t（m3），按 net.branches() 的阀门顺序。"""
    return {v: round(float(quota * a), 1) for (v, _, _), a in zip(net.branches(), area)}


def pipe_inventory():
    """可发生爆管或泄漏的管段：(管段名, 类别, 长度, 内径 mm, 下游节点, 所属分干管 x 或 None)。"""
    out = [(f"M{i}", "main", net.MAIN_LEN[i], net.D_MAIN, f"J{i}", None) for i in range(10)]
    for x in range(10):
        for k in range(1, 7):
            out.append((f"P{x}{k}", "sub", net.SUB_LEN[k - 1], net.D_SUB, f"J{x}{k}", x))
            for s in net.SIDES:
                out.append((f"PB{x}{k}{s}", "branch", net.BRANCH_LEN - net.STUB, net.D_BRANCH, f"E{x}{k}{s}", x))
    return out


def isolation(pipe):
    """爆管隔离阀和被隔离的支管阀。分干管段没有中间阀，只能关首端阀 VSx，整条分干管退出。"""
    if pipe.startswith("PB"):
        v = f"V{pipe[2:]}"
        return [v], [v]
    x = pipe[1]
    return [f"VS{x}"], [f"V{x}{k}{s}" for k in range(1, 7) for s in net.SIDES]


def sample_fault(rng, ftype, holdout):
    """holdout=True 只在分干管 J7 上抽位置，False 只在其他位置抽。"""
    def ok(x):
        return (x == HOLDOUT_SUBMAIN) == holdout

    if ftype in ("burst", "leak"):
        cand = [p for p in pipe_inventory() if (p[1] != "main" or ftype == "leak")
                and (ok(p[5]) if p[5] is not None else not holdout)]
        w = np.array([p[2] for p in cand]); w /= w.sum()
        name, cls, L, d, node, x = cand[rng.choice(len(cand), p=w)]
        f = dict(type=ftype, pipe=name, pipe_class=cls, submain=x, node=node, pos_frac=round(float(rng.uniform()), 3))
        if ftype == "burst":
            frac = rng.uniform(*BURST_AREA_FRAC[cls])
            area = frac * np.pi * (d / 1000) ** 2 / 4
            f.update(area_frac=round(float(frac), 3),
                     emitter_coeff=round(float(CD * area * np.sqrt(2 * 9.81)), 6),   # Q = C·sqrt(p)，m3/s, m
                     isolate_valves=isolation(name)[0], removed_valves=isolation(name)[1])
        else:
            q = rng.uniform(*LEAK_Q_M3H)
            f.update(q_ref_m3h=round(float(q), 2), p_ref_m=15.0,
                     emitter_coeff=round(float(q / 3600 / np.sqrt(15.0)), 7))
        return f
    valves = [v for v, *_ in net.branches() if ok(int(v[1]))] + [f"VS{x}" for x in range(10) if ok(x)]
    v = valves[rng.integers(len(valves))]
    mode = rng.choice(["stuck_open", "stuck_closed", "stuck_partial"])
    f = dict(type="stuck", valve=v, submain=int(v[-1] if v.startswith("VS") else v[1]), mode=str(mode))
    if mode == "stuck_partial":
        f["opening"] = round(float(rng.uniform(0.2, 0.8)), 2)
    return f


def sample_timing(rng):
    if rng.uniform() < 0.7:
        return dict(phase="running", cycle_frac=round(float(rng.uniform()), 4))
    return dict(phase="switching", switch_frac=round(float(rng.uniform()), 4),
                t_in_switch_s=round(float(rng.uniform(0, 60)), 1))


def make_scenario(rng, sid, kind, event, delta, holdout_loc=False, combined=False):
    order, quota, start, end = next(e for e in IRRIGATION_EVENTS if e[0] == event)
    seed = int(rng.integers(2 ** 31))
    r = np.random.default_rng(seed)
    pert = dict(delta=delta, seed=seed,
                roughness_mult=round(float(1 + r.uniform(-delta, delta)), 3),
                wavespeed_mult=round(float(1 + r.uniform(-delta, delta)), 3))
    area = branch_area(r, delta)
    sc = dict(id=sid, kind=kind, irrigation_event=order, quota_m3_per_mu=quota, window=f"{start}~{end}",
              perturbation=pert, demand_m3=field_demand(quota, area))
    if kind == "fault":
        types = ["burst", "leak", "stuck"]
        n = 2 if combined else 1
        picks = rng.choice(types, size=n, replace=not combined)
        sc["faults"] = [dict(**sample_fault(rng, str(t), holdout_loc), **sample_timing(rng)) for t in picks]
        if combined and rng.uniform() < 0.5:        # 先后发生：第二个故障晚于第一个
            sc["faults"][1]["after_first"] = True
    return sc


def split_of(sc):
    ood = []
    if sc["irrigation_event"] == HOLDOUT_EVENT: ood.append("holdout_demand")
    if sc["perturbation"]["delta"] >= DELTA_OOD: ood.append("max_perturbation")
    if sc["kind"] == "fault":
        if any(f["submain"] == HOLDOUT_SUBMAIN for f in sc["faults"]):
            ood.append("holdout_location")
        if len(sc["faults"]) > 1: ood.append("combined_fault")
    return ood


def generate(n_fault=1200, n_ood=300, seed=2026):
    rng = np.random.default_rng(seed)
    in_events = [e[0] for e in IRRIGATION_EVENTS if e[0] != HOLDOUT_EVENT]
    out = []
    for kind in ("nominal", "fault"):
        for i in range(n_fault):
            sc = make_scenario(rng, f"{kind[0].upper()}{i:05d}", kind, int(rng.choice(in_events)),
                               float(rng.choice(DELTAS_IN)))
            u = rng.uniform()
            sc["split"] = "train" if u < 0.70 else "val" if u < 0.85 else "test"
            out.append(sc)
    # 分布外：四类各占四分之一（标称场景只取留出需水和最大扰动两类）
    for i in range(n_ood):
        c = i % 4
        kind = "nominal" if (c < 2 and i % 8 < 4) else "fault"
        sc = make_scenario(rng, f"O{i:05d}", kind,
                           HOLDOUT_EVENT if c == 0 else int(rng.choice(in_events)),
                           DELTA_OOD if c == 1 else float(rng.choice(DELTAS_IN)),
                           holdout_loc=(c == 2), combined=(c == 3))
        sc["split"] = "ood"
        out.append(sc)
    for sc in out:
        sc["ood_tags"] = split_of(sc)
        assert (sc["split"] == "ood") == bool(sc["ood_tags"]), sc["id"]
    return out


if __name__ == "__main__":
    import collections
    S = generate()
    with open(HERE / "scenarios.jsonl", "w", encoding="utf-8") as f:
        for sc in S:
            f.write(json.dumps(sc, ensure_ascii=False) + "\n")
    print("场景总数", len(S))
    print(collections.Counter((s["split"], s["kind"]) for s in S))
    print(collections.Counter(f["type"] for s in S if s["kind"] == "fault" for f in s["faults"]))
    print(collections.Counter(t for s in S for t in s["ood_tags"]))
    D = np.array([list(s["demand_m3"].values()) for s in S])
    print("支管需水 D_i 范围 %.0f–%.0f m3（47 m3/h 时 %.1f–%.1f h）" % (D.min(), D.max(), D.min() / 47, D.max() / 47))
