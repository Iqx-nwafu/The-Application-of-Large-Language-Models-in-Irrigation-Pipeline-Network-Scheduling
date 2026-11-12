"""瞬态终检：把一次切换（或故障处置动作）放进 moc.py 全量计算，逐条检查瞬态约束。

承压分区：干管、分干管、阀前短管和阀前节点（M*, P*, PA*, PC*, J*, A*, S*, C*）按 PVC PN 0.6 MPa；
支管阀后（PB*, B*, E*）按 PE PN 0.25 MPa。α = 1.0。
p_min ≥ H_vac 只校核阀前的 PVC 部分。支管阀后是带滴灌带的多孔管，沿程进气，模型把出流集中在末端，
阀后负压是集中出流假设的数值结果（与 TSNet 相同），只记录不作约束。
动作历时：支管阀 ≥ 5 s，分干管阀 ≥ 10 s（instant 只允许出现在 emergency=True 的故障隔离中）。
"""
import numpy as np
from moc import TransientModel

ALPHA, PN_PVC, PN_PE, H_VAC = 1.0, 61.2, 25.5, -5.0
TAU_MIN = {"branch": 5.0, "sub": 10.0}
SETTLE_S = 15.0

_sims = {}


def sim_for(ctx):
    pert = getattr(ctx, "pert", {}) or {}
    key = (round(pert.get("roughness_mult", 1.0), 4), round(pert.get("wavespeed_mult", 1.0), 4), pert.get("h_source"))
    if key not in _sims:
        _sims[key] = TransientModel(rough_mult=key[0], wavespeed_mult=key[1])
        if key[2] is not None:                       # 高水头场景：瞬态内核和它自带的稳态一起改水源水头
            _sims[key].H_res = _sims[key].steady.H0 = float(key[2])
    return _sims[key]


def isolated_zone(valves):
    """关闭这些隔离阀后退出运行的管段和节点（阀后侧）。难瞬态子集里隔离段在关阀后泄空，不作瞬态约束。"""
    out = set()
    for v in valves:
        if v.startswith("VS"):
            x = v[2]
            out |= {f"S{x}"} | {f"J{x}{k}" for k in range(1, 7)} | {f"P{x}{k}" for k in range(1, 7)}
            out |= {f"{p}{x}{k}{s}" for k in range(1, 7) for s in "LR" for p in ("C", "B", "E", "PC", "PB")}
        else:
            out |= {f"B{v[1:]}", f"E{v[1:]}", f"PB{v[1:]}"}
    return out


def is_pe(name):
    return name[0] in "BE" or name.startswith("PB")


def action_end(a):
    return a["t_start_s"] + (1.0 if a["template"] == "instant" else a["params"]["duration_s"])


def tau_violations(actions, emergency=False):
    out = []
    for a in actions:
        d = 1.0 if a["template"] == "instant" else a["params"]["duration_s"]
        cls = "sub" if a["valve"].startswith("VS") else "branch"
        if a["template"] == "instant" and not emergency:
            out.append((a["valve"], "instant_not_allowed", d))
        elif d < TAU_MIN[cls] and not (emergency and a["template"] == "instant"):
            out.append((a["valve"], "tau_min", round(TAU_MIN[cls] - d, 2)))
    return out


def simulate_switch(ctx, open_before, actions, events=(), tf=None):
    """open_before: 切换前开启的支管阀（含卡开）；actions: schema 的 switch.actions。"""
    sim = sim_for(ctx)
    tf = tf or (max([action_end(a) for a in actions] + [e[2] + e[3] for e in events] + [0.0]) + SETTLE_S)
    iso = getattr(ctx, "iso_valves", set())
    if iso:                                          # 难瞬态子集：爆管出流中、隔离阀开着起算；爆管移除的支管阀可操作
        em = dict(ctx.emitters)
        for n, c in ctx.iso_emitters.items():
            em[n] = em.get(n, 0) + c
        out = sim.run(sorted(open_before), actions, tf=tf, emitters=em, openings=ctx.openings,
                      closed_links=sorted(set(ctx.closed_links) - iso), events=events, stuck=set(ctx.always_open))
        out["exclude"] = isolated_zone({a["valve"] for a in actions if a["op"] == "close"} & iso)
        return out
    stuck = {v for v in ctx.always_open} | set(ctx.removed & {a["valve"] for a in actions})
    return sim.run(sorted(open_before), actions, tf=tf, emitters=ctx.emitters, openings=ctx.openings,
                   closed_links=sorted(ctx.closed_links), events=events, stuck=stuck)


def check_transient(out):
    """返回 (violations, extremes)。violations: [{type, where, value, limit}]。out["exclude"]（难瞬态子集的隔离段）不校核。"""
    viol = []
    ex = out.get("exclude") or ()
    pmax = {n: v for n, v in {**out["p_max"], **out["pipe_p_max"]}.items() if n not in ex}
    pmin = {n: v for n, v in {**out["p_min"], **out["pipe_p_min"]}.items() if n not in ex}
    for n, v in pmax.items():
        if not np.isfinite(v):
            continue
        lim = ALPHA * (PN_PE if is_pe(n) else PN_PVC)
        if v > lim:
            viol.append(dict(type="p_max", where=n, value=round(float(v), 2), limit=lim))
    for n, v in pmin.items():
        if np.isfinite(v) and v < H_VAC and not is_pe(n):
            viol.append(dict(type="p_min", where=n, value=round(float(v), 2), limit=H_VAC))
    pvc = [v for n, v in pmax.items() if not is_pe(n) and np.isfinite(v)]
    ext = dict(p_max_pvc=round(float(max(pvc)), 2), p_max_ratio=round(float(max(pvc)) / PN_PVC, 3),
               p_min=round(float(min(v for n, v in pmin.items() if np.isfinite(v) and not is_pe(n))), 2),
               p_min_branch=round(float(min(v for n, v in pmin.items() if np.isfinite(v) and is_pe(n))), 2),
               below_vapor=out["below_vapor"])
    viol.sort(key=lambda d: -abs(d["value"] - d["limit"]))
    return viol, ext


def safety_summary(out):
    """难瞬态子集的安全口径（只给评测记录，工具返回给模型的内容不变）：
    最大水锤压力（PVC、PE 分区）、相对承压上限的最坏比、最低压力、越限点数（按分区和类型）。"""
    viol, ext = check_transient(out)
    ex = out.get("exclude") or ()
    pmax = {n: v for n, v in {**out["p_max"], **out["pipe_p_max"]}.items() if n not in ex and np.isfinite(v)}
    pe = max(v for n, v in pmax.items() if is_pe(n))
    return dict(n_viol=len(viol), n_pmax_pvc=sum(d["type"] == "p_max" and not is_pe(d["where"]) for d in viol),
                n_pmax_pe=sum(d["type"] == "p_max" and is_pe(d["where"]) for d in viol),
                n_pmin=sum(d["type"] == "p_min" for d in viol), p_max_pvc=ext["p_max_pvc"], p_max_pe=round(float(pe), 2),
                p_min=ext["p_min"], worst_ratio=round(max(ext["p_max_pvc"] / (ALPHA * PN_PVC), pe / (ALPHA * PN_PE)), 3),
                vac_margin=round(ext["p_min"] - H_VAC, 2))
