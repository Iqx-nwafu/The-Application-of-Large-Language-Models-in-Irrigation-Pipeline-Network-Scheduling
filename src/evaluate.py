"""轮灌计划的稳态评价：工况上下文（需水、扰动、故障）+ 目标分项 + 逐条约束。

GA/NSGA-II、check_constraints 工具和安全兜底层共用这里的口径（A1 §3、§3a）：
  q_i     支管实际流量，快速稳态求解（滴头 x = 0.5）
  T_k     = λ·max_{i∈k} D_i/q_i，λ = min(1, (T_win − 切换时间)/Σ max D/q)，T_k 截到 [0.5, 8] h
  D_def   = Σ max(0, D_i − q_i·T_k)
  CV_q    灌溉支管实际流量的变异系数
  V_over  = Σ max(0, q_i·T_k − D_i) 超灌水量；E_pump = ρg·H0·Σ Q_hub,k·T_k（kWh，定水头下与 V_over 线性相关，只报告）
  标量化按 scoring.py（方案 A）：缺水字典序优先，其余熵权 TOPSIS 贴近度 C*（score）
  支管入口超 P_B_CAP 的阀由稳态自动节流，开度在 per_group[k]["openings"]
卡开支管阀：并入同一分干管有支管轮灌的组，其余组关 VSx 截断；部分开度卡死同理。
约束：Q_hub ≤ 195，支管末端 p ≥ 10 m，干管/分干管 v ≤ 2.5 m/s，稳态 p ≤ PN（PE 支管 ≤ PN − 3 m 瞬态裕量），Σ T_k + 切换 ≤ 120 h（判定容差 T_WIN_TOL），
      每组 1–8 个阀，每个可用阀恰好一组，被隔离或卡关的阀不得入组。
"""
import numpy as np
import network as net
from hydraulics import SteadyModel, P_B_CAP
import scoring

Q_MAX, H_MIN, V_MAX, T_WIN = 195.0, 10.0, 2.5, 120.0
PN_PVC, PN_PE = 61.2, 25.5
PE_MARGIN = 3.0                        # 支管入口稳态留给瞬态的裕量，α 仍为 1.0（作者 2026-10-27）
SWITCH_H = 2 / 60                      # 每次切换计 2 min（错开动作 + 两阶段关阀）
T_RANGE = (0.5, 8.0)
# 时间窗判定容差（h）。方案里各组时长按 0.001 h 记录，40 余组舍入后求和可比 120 h 多出约 0.001–0.005 h，
# 论文对所有方法统一按 0.03 h（不足 2 min）判定。实验运行时闭环内用的是 1e-6；设 T_WIN_TOL = 1e-6 即还原当时的判定。
T_WIN_TOL = 0.03
ALL_VALVES = [v for v, *_ in net.branches()]

_models = {}


def model(rough_mult=1.0, h_source=None):
    """h_source：水源水头（难瞬态子集的高水头场景，perturbation.h_source）；None 为 INP 里的 25 m。"""
    key = (round(rough_mult, 4), h_source)
    if key not in _models:
        _models[key] = SteadyModel(rough_mult=rough_mult)
        if h_source is not None:
            _models[key].H0 = float(h_source)
    return _models[key]


class Context:
    """一次调度决策面对的工况。fault_response 时把故障翻译成稳态边界条件。"""

    def __init__(self, scenario=None, apply_faults=True):
        sc = scenario or {}
        self.id = sc.get("id", "baseline")
        self.D = sc.get("demand_m3") or {v: 860 / 120 * 25 for v in ALL_VALVES}
        self.pert = sc.get("perturbation", {})
        self.m = model(self.pert.get("roughness_mult", 1.0), self.pert.get("h_source"))
        self.removed, self.always_open, self.openings, self.closed_links, self.emitters = set(), set(), {}, set(), {}
        # 难瞬态子集（sc["iso_transient"]）：瞬态校核从"爆管出流中、隔离阀开着"的稳态起算，关隔离阀本身的水锤计入；
        # 主实验口径（默认）是隔离阀在初始稳态里已关闭，隔离动作不产生瞬态。稳态评价两种口径相同（按已隔离算）。
        self.iso_valves, self.iso_emitters = set(), {}
        self.t_win = T_WIN
        for f in (sc.get("faults", []) if apply_faults else []):
            if f["type"] == "burst":
                self.closed_links |= set(f["isolate_valves"]); self.removed |= set(f["removed_valves"])
                if sc.get("iso_transient"):
                    self.iso_valves |= set(f["isolate_valves"])
                    self.iso_emitters[f["node"]] = self.iso_emitters.get(f["node"], 0) + f["emitter_coeff"]
            elif f["type"] == "leak":
                self.emitters[f["node"]] = self.emitters.get(f["node"], 0) + f["emitter_coeff"]
            elif f["mode"] == "stuck_closed":
                v = f["valve"]
                if v.startswith("VS"):
                    self.closed_links.add(v); self.removed |= {u for u in ALL_VALVES if u[1] == v[2]}
                else:
                    self.removed.add(v)
            elif f["mode"] == "stuck_open" and not f["valve"].startswith("VS"):
                self.always_open.add(f["valve"])
            elif f["mode"] == "stuck_partial":
                self.openings[f["valve"]] = f["opening"]
                if not f["valve"].startswith("VS"):
                    self.always_open.add(f["valve"])
        self.valves = [v for v in ALL_VALVES if v not in self.removed and v not in self.always_open]

    def stuck_active(self, g):
        """卡开支管只在本分干管有支管轮到时出水，其余组关分干管首端阀 VSx 截断。"""
        xs = {v[1] for v in g}
        return {v for v in self.always_open if v[1] in xs}

    def solve(self, groups):
        sets = [set(g) | self.stuck_active(g) for g in groups]
        cut = [{f"VS{v[1]}" for v in self.always_open if v[1] not in {u[1] for u in g}} for g in groups]
        return self.m.solve(sets, extra_emitters=[self.emitters] * len(sets), openings=self.openings,
                            closed_links=self.closed_links, closed_per_set=cut)


def schedule(ctx, groups, r):
    """由实际流量定 T_k（h），并返回各支管的灌水量。"""
    vi = {v: i for i, v in enumerate(ctx.m.valves)}
    need = np.array([max(ctx.D[v] / max(r["q_branch"][k, vi[v]], 1e-6) for v in g) for k, g in enumerate(groups)])
    t_avail = ctx.t_win - SWITCH_H * (len(groups) - 1)
    lam = min(1.0, t_avail / need.sum()) if need.sum() > 0 else 1.0
    return np.clip(need * lam, *T_RANGE)


def evaluate(ctx, groups, durations=None):
    """groups: 阀门列表的列表（不含卡开阀，卡开阀自动并入每组）；durations: 给定 T_k（h），None 时按 schedule 定。
    返回 dict(objectives, J, violations, per_group)。violations 为 {约束: 最大越限量}，空表示可行。"""
    viol = {}
    flat = [v for g in groups for v in g]
    bad = [v for v in flat if v in ctx.removed]
    if bad: viol["isolated_valve_assigned"] = len(bad)
    missing = set(ctx.valves) - set(flat)
    if missing: viol["valve_unassigned"] = len(missing)
    dup = len(flat) - len(set(flat))
    if dup: viol["valve_duplicated"] = dup
    sizes = [len(g) for g in groups]
    if max(sizes) > 8 or min(sizes) < 1: viol["group_size"] = max(max(sizes) - 8, 1 - min(sizes))

    r = ctx.solve(groups)
    T = schedule(ctx, groups, r) if durations is None else np.asarray(durations, float)
    m, vi = ctx.m, {v: i for i, v in enumerate(ctx.m.valves)}
    irrigated, vol, q_all, pe_min = {}, {}, [], []
    for k, g in enumerate(groups):
        for v in list(g) + sorted(ctx.stuck_active(g)):
            q = r["q_branch"][k, vi[v]]
            vol[v] = vol.get(v, 0.0) + q * T[k]
            if v in g:
                q_all.append(q)
        pe_min.append(np.nanmin(r["p"][k, m.e_idx[[vi[v] for v in g]]]))
    deficit = sum(max(0.0, ctx.D[v] - vol.get(v, 0.0)) for v in ctx.valves + sorted(ctx.always_open))
    over = sum(max(0.0, vol.get(v, 0.0) - ctx.D[v]) for v in ctx.valves + sorted(ctx.always_open))
    q_all = np.array(q_all)
    cv = float(q_all.std() / q_all.mean()) if len(q_all) else 0.0
    t_total = float(T.sum() + SWITCH_H * (len(groups) - 1))
    energy = float(1000 * 9.81 * m.H0 * (r["Q_hub"] / 3600 * T * 3600).sum() / 3.6e6)

    pe_min = np.array(pe_min)
    b_idx = np.array([m.idx[f"B{v[1:]}"] for v in m.valves])
    pmax_pe = np.nanmax(r["p"][:, np.r_[m.e_idx, b_idx]])            # 阀后 PE 支管：B（入口）和 E（末端）
    pmax_all = np.nanmax(r["p"])
    vmax = r["v"][:, np.r_[m.main, m.sub]].max(axis=1)
    checks = {"Q_hub": (r["Q_hub"] - Q_MAX).max(), "H_min": (H_MIN - pe_min).max(),
              "v_max": (vmax - V_MAX).max(), "PN_PE": pmax_pe - (PN_PE - PE_MARGIN) - 0.02,      # 自动节流收敛到 0.005 m，留数值容差
              "PN_PVC": pmax_all - PN_PVC,       # 实验运行版本里这一项被上一行的行内注释吞掉；全部场景水源水头 ≤ 35 m < 61.2 m，不影响任何结果
              "T_window": t_total - ctx.t_win}
    viol.update({k: round(float(v), 3) for k, v in checks.items() if v > (T_WIN_TOL if k == "T_window" else 1e-6)})

    D_sum = sum(ctx.D[v] for v in ctx.valves)
    obj = dict(deficit_m3=round(deficit, 1), cv_q=round(cv, 4), T_total_h=round(t_total, 2), over_m3=round(over, 1),
               E_kWh=round(energy, 1))
    score = scoring.closeness(obj)                              # 方案 A 的贴近度 C*；没有参考集时为 None
    per_group = [dict(valves=list(g), duration_h=round(float(T[k]), 3), Q_hub=round(float(r["Q_hub"][k]), 1),
                      p_end_min=round(float(pe_min[k]), 2), v_max=round(float(vmax[k]), 2),
                      openings={v: round(float(r["opening"][k, vi[v]]), 2) for v in g if r["opening"][k, vi[v]] < 0.995})
                 for k, g in enumerate(groups)]
    return dict(objectives=obj, score=score, feasible=not viol, violations=viol, per_group=per_group, D_total=round(D_sum, 1))

def to_rotation_plan(result):
    groups = [dict(group_id=k + 1, valves=g["valves"], duration_h=g["duration_h"]) for k, g in enumerate(result["per_group"])]
    return dict(groups=groups, order=[g["group_id"] for g in groups])


if __name__ == "__main__":
    ctx = Context()
    res = evaluate(ctx, net.rotation_groups(), durations=[4.0] * 30)
    print("基线轮灌表（30 组 × 4 h，定额 25 m3/亩）")
    print(res["objectives"], "C* =", res["score"])
    thr = {v: s for g in res["per_group"] for v, s in g["openings"].items()}
    print("自动节流的支管阀", len(thr), "个")
    print("越限：", res["violations"])
