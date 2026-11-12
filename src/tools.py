"""A7 工作流工具接口 + 安全兜底层。决策模型（API 大模型或轻量化模型）只通过这些函数和仿真器交互。

工具（输入输出都是可 JSON 序列化的 dict，见 TOOL_SPECS）
  simulate_steady      轮灌计划 → 各组压力、流速、首部流量、自动节流开度、目标分项与 C*、稳态越限
  screen_transient     切换动作 → 包络库估计的阀前 PVC 区极值（毫秒级，阀后 PE 区不覆盖）
  simulate_transient   切换或故障处置动作 → 全量特征线法极值和越限（约 0.3–0.6 s）
  check_constraints    完整决策 JSON → 逐条约束（结构、τ_min、稳态、瞬态）
  query_node           节点、管段或阀门 → 高程、管径、承压、所属轮灌组
  run_optimizer        NSGA-II 轮灌分组（约 30 s，22 进程）

安全兜底层（独立于决策模型）
  project_action       参数投影到模板库离散档，且不快于 τ_min
  safe_switch          规则切换（先开后关、错开、两阶段关），全量校核不过就逐档放慢，最后逐阀串行
  fallback_rotation    run_optimizer，失败时用预设轮灌表
"""
import json
import pathlib
import time
import numpy as np
import network as net
import evaluate as ev
import transient_tools as tt
import envelope as envmod

HERE = pathlib.Path(__file__).parent
_env = None


def _envelope():
    global _env
    if _env is None:
        _env = envmod.Envelope(margin=json.load(open(HERE / "envelope_margin.json", encoding="utf-8"))["pvc"]["margin_m"])
    return _env


def _groups(plan):
    order = plan.get("order") or [g["group_id"] for g in plan["groups"]]
    by_id = {g["group_id"]: g for g in plan["groups"]}
    gs = [by_id[i] for i in order]
    return [g["valves"] for g in gs], [g.get("duration_h") for g in gs]


# ---------------- 工具 ----------------
def simulate_steady(ctx, rotation_plan, given_durations=True):
    groups, T = _groups(rotation_plan)
    r = ev.evaluate(ctx, groups, durations=T if given_durations and None not in T else None)
    return dict(objectives=r["objectives"], score=r["score"], feasible=r["feasible"], violations=r["violations"],
                per_group=r["per_group"])


def screen_transient(ctx, open_before, actions):
    env = _envelope()
    sim = tt.sim_for(ctx)
    after = (set(open_before) - {a["valve"] for a in actions if a["op"] == "close"}) | \
            {a["valve"] for a in actions if a["op"] == "open"}
    if ctx.iso_valves:                              # 难瞬态子集：切换前爆管出流、隔离阀开着；切换后按动作关上的隔离阀算
        iso_closed = {a["valve"] for a in actions if a["op"] == "close"} & ctx.iso_valves
        sb = envmod.steady_p(sim, open_before, sorted(set(ctx.closed_links) - ctx.iso_valves), ctx.iso_emitters)
        sa = envmod.steady_p(sim, after, sorted(set(ctx.closed_links) - ctx.iso_valves | iso_closed), ctx.iso_emitters)
    else:
        sb, sa = envmod.steady_p(sim, open_before, sorted(ctx.closed_links)), envmod.steady_p(sim, after, sorted(ctx.closed_links))
    acts = [project_action(a) for a in actions]
    pmax, pmin = env.screen(sb, sa, acts, open_before)
    vmax, vmin = env.violations(pmax, pmin)
    pvc = ~env.pe
    flagged = [dict(type="p_max", where=env.locs[j], est=round(float(pmax[j]), 1)) for j in np.where((vmax > 0) & pvc)[0]]
    flagged += [dict(type="p_min", where=env.locs[j], est=round(float(pmin[j]), 1)) for j in np.where((vmin > 0) & pvc)[0]]
    return dict(pass_screen=not flagged, est_p_max_pvc=round(float(pmax[pvc].max()), 1),
                est_p_min_pvc=round(float(np.nanmin(pmin[pvc])), 1), flagged=flagged[:10],
                note="包络库只覆盖阀前 PVC 区；阀后 PE 支管的升压须全量计算")


def simulate_transient(ctx, open_before, actions, events=()):
    t0 = time.time()
    out = tt.simulate_switch(ctx, open_before, [project_action(a) for a in actions], events=events)
    viol, ext = tt.check_transient(out)
    return dict(feasible=not viol, violations=viol[:10], n_violations=len(viol), extremes=ext,
                cpu_s=round(time.time() - t0, 2))


def check_constraints(ctx, decision, open_before=None, events=(), skip_transient=False):
    """decision: decision_output.schema.json 的 JSON。switch 需给出切换前开启的阀门（默认取 from_group 的阀）。
    skip_transient=True 只做结构、τ_min 和稳态（消融"去掉瞬态校核"用；评测时最终仍用全量校核）。"""
    out = dict(decision_type=decision["decision_type"], checks={})
    c = out["checks"]
    if decision["decision_type"] in ("rotation_plan", "fault_response"):
        plan = decision["rotation_plan"] if decision["decision_type"] == "rotation_plan" else decision["fault_response"]["remaining_plan"]
        s = simulate_steady(ctx, plan)
        c["steady"] = dict(feasible=s["feasible"], violations=s["violations"], objectives=s["objectives"])
    acts = decision.get("switch", {}).get("actions") or decision.get("fault_response", {}).get("isolation_actions") or []
    if ctx.iso_valves and decision["decision_type"] == "fault_response":   # 难瞬态子集：隔离阀必须由动作关上（否则爆管未隔离）
        last = {a["valve"]: a["op"] for a in sorted(acts, key=lambda a: a.get("t_start_s", 0))}
        miss = sorted(v for v in ctx.iso_valves if last.get(v) != "close")
        c["isolation"] = dict(feasible=not miss, missing_isolation_valves=miss)
    if acts:
        emergency = decision["decision_type"] == "fault_response"
        tv = tt.tau_violations(acts, emergency=emergency)
        c["tau_min"] = dict(feasible=not tv, violations=[dict(valve=v, type=k, value=x) for v, k, x in tv])
        if not skip_transient:
            c["transient"] = simulate_transient(ctx, open_before or [], acts, events)
    out["feasible"] = all(v["feasible"] for v in c.values())
    return out


def query_node(name, rotation_plan=None):
    sim = tt.sim_for(type("C", (), {"pert": {}})())
    wn = sim.wn
    info = dict(name=name)
    if name in wn.node_name_list:
        n = wn.get_node(name)
        info.update(kind=n.node_type, elevation_m=round(n.elevation, 2) if hasattr(n, "elevation") else None,
                    pn_m=tt.PN_PE if tt.is_pe(name) else tt.PN_PVC)
    elif name in wn.link_name_list:
        l = wn.get_link(name)
        info.update(kind=l.link_type, diameter_mm=round(l.diameter * 1000, 1), from_node=l.start_node_name,
                    to_node=l.end_node_name, length_m=getattr(l, "length", None),
                    pn_m=tt.PN_PE if tt.is_pe(name) else tt.PN_PVC,
                    material="PE" if name.startswith(("PB", "PC")) else "PVC-U")
        if l.link_type == "Valve":
            info["tau_min_s"] = tt.TAU_MIN["sub" if name.startswith("VS") else "branch"]
    else:
        return dict(name=name, error="未知编号")
    if rotation_plan:
        info["group_id"] = next((g["group_id"] for g in rotation_plan["groups"] if name in g["valves"]), None)
    return info


def run_optimizer(ctx, scenario, apply_faults=True, gens=150):
    """按当前决策上下文优化：已灌完的支管不参与，时间窗取剩余时间。"""
    import optimizer
    exclude = [v for v in ev.ALL_VALVES if v not in ctx.valves]
    r = optimizer.run_optimizer(scenario, apply_faults, exclude=exclude, t_win=ctx.t_win, gens=gens)
    return dict(rotation_plan=r["rotation_plan"], objectives=r["best"]["objectives"], score=r["best"]["score"],
                feasible=r["best"]["feasible"], violations=r["best"]["violations"], cpu_s=r["cpu_s"])


# ---------------- 安全兜底层 ----------------
def project_action(a):
    """把参数投影到模板库离散档：同模板中历时不短于请求值的最快档（不会比模型要的更快）；不低于 τ_min。"""
    a = dict(a)
    cls = "sub" if a["valve"].startswith("VS") else "branch"
    if a["template"] == "instant":
        return a
    temps = [p for t, p in envmod.GRID[cls][a["op"]] if t == a["template"]]
    if not temps:                                   # 分干管阀没有两阶段档：换成同历时的线性
        a["template"] = f"{a['op']}_linear"
        temps = [p for t, p in envmod.GRID[cls][a["op"]] if t == a["template"]]
    d = max(float(a["params"].get("duration_s", 0)), tt.TAU_MIN[cls])
    slower = [p for p in temps if p["duration_s"] >= d]
    a["params"] = dict(min(slower, key=lambda p: p["duration_s"]) if slower else max(temps, key=lambda p: p["duration_s"]))
    return a


def rule_switch(g1, g2, level=0):
    """工程规则切换。level 越大越慢：开阀、关阀模板逐档放慢，错开时间加大；level ≥ 4 逐阀串行。"""
    opens = [("open_linear", dict(duration_s=10)), ("open_linear", dict(duration_s=20)),
             ("open_two_stage", dict(duration_s=30, break_time_s=10, break_opening=0.3)),
             ("open_linear", dict(duration_s=40)), ("open_linear", dict(duration_s=40))]
    closes = [("close_two_stage", dict(duration_s=15, break_time_s=3, break_opening=0.3)),
              ("close_two_stage", dict(duration_s=20, break_time_s=5, break_opening=0.3)),
              ("close_two_stage", dict(duration_s=30, break_time_s=5, break_opening=0.2)),
              ("close_two_stage", dict(duration_s=40, break_time_s=10, break_opening=0.2)),
              ("close_two_stage", dict(duration_s=40, break_time_s=10, break_opening=0.2))]
    lv = min(level, 4)
    gap = [2, 3, 5, 8, 0][lv]
    acts, t = [], 0.0
    for v in sorted(g2):                                  # 先开
        tmpl, p = opens[lv]
        acts.append(dict(valve=v, op="open", t_start_s=t, template=tmpl, params=p))
        t += gap if lv < 4 else p["duration_s"] + 5
    t = (acts[-1]["t_start_s"] + 4 if lv < 4 else t) if acts else 0.0
    for v in sorted(g1, key=lambda v: int(v[2])):         # 后关，分干管首端先关，末端（k=5,6）放到最后
        tmpl, p = closes[min(lv + (1 if int(v[2]) >= 5 else 0), 4)]
        acts.append(dict(valve=v, op="close", t_start_s=t, template=tmpl, params=p))
        t += gap if lv < 4 else p["duration_s"] + 5
    return acts


def safe_switch(ctx, g1, g2, max_level=4):
    """兜底切换：从规则方案开始，全量校核不过就放慢一档。返回 (actions, 检查结果, 用到的档位)。"""
    for lv in range(max_level + 1):
        acts = rule_switch(g1, g2, lv)
        r = simulate_transient(ctx, set(g1) | ctx.stuck_active(g1), acts)
        if r["feasible"]:
            return acts, r, lv
    return acts, r, lv


ISO_LEVELS = {"sub": [("close_linear", dict(duration_s=d)) for d in (10, 20, 40, 60)],      # safe_fault 逐档放慢（模板库离散档）
              "branch": [("close_linear", dict(duration_s=5)), ("close_two_stage", dict(duration_s=15, break_time_s=3, break_opening=0.3)),
                         ("close_two_stage", dict(duration_s=30, break_time_s=5, break_opening=0.2)),
                         ("close_two_stage", dict(duration_s=40, break_time_s=10, break_opening=0.2))]}


def rule_fault(ctx, scenario, plan=None, level=0):
    """规则故障处置：爆管关隔离阀（level 0 = 规范最短历时线性关），泄漏和卡死不隔离；剩余支管用预设轮灌表（或给定 plan）。"""
    f = scenario["faults"][0]
    iso = list(f.get("isolate_valves", [])) if f["type"] == "burst" else []
    acts = [dict(valve=v, op="close", t_start_s=0.0, template=t, params=dict(p))
            for v in iso for t, p in [ISO_LEVELS["sub" if v.startswith("VS") else "branch"][min(level, 3)]]]
    if plan is None:
        groups = [[v for v in g if v in ctx.valves] for g in net.rotation_groups()]
        groups = [g for g in groups if g]
        plan = dict(groups=[dict(group_id=k + 1, valves=g, duration_h=4.0) for k, g in enumerate(groups)],
                    order=list(range(1, len(groups) + 1)))
    return dict(schema_version="0.1", scenario_id=scenario["id"], decision_type="fault_response", reasoning="工程规则：隔离 + 预设轮灌表",
                fault_response=dict(fault_id=scenario["id"], isolate_valves=iso, isolation_actions=acts,
                                    removed_valves=sorted(ctx.removed), remaining_plan=plan))


def safe_fault(ctx, scenario, plan=None, open_before=()):
    """兜底故障处置（难瞬态子集，隔离关阀计入瞬态）：规则处置从规范最短历时开始，全量校核不过就放慢一档。
    非难子集场景（ctx.iso_valves 为空）等同 rule_fault。返回 (决策, 用到的档位)。"""
    d = rule_fault(ctx, scenario, plan)
    if not ctx.iso_valves:
        return d, 0
    for lv in range(4):
        d = rule_fault(ctx, scenario, plan, lv)
        if simulate_transient(ctx, open_before, d["fault_response"]["isolation_actions"])["feasible"]:
            return d, lv
    return d, lv


def fallback_rotation(ctx, scenario):
    try:
        r = run_optimizer(ctx, scenario, apply_faults=bool(scenario and scenario.get("faults")))
        if r["feasible"]:
            return r["rotation_plan"], "optimizer"
    except Exception:
        pass
    groups = [[v for v in g if v in ctx.valves] for g in net.rotation_groups()]
    groups = [g for g in groups if g]
    plan = dict(groups=[dict(group_id=k + 1, valves=g, duration_h=4.0) for k, g in enumerate(groups)],
                order=list(range(1, len(groups) + 1)))
    return plan, "preset_table"


TOOL_SPECS = [
    {"name": "simulate_steady", "description": "对轮灌计划做稳态计算，返回各组首部流量、支管末端最低压力、流速、目标分项和稳态越限",
     "parameters": {"type": "object", "properties": {"rotation_plan": {"type": "object"}}, "required": ["rotation_plan"]}},
    {"name": "screen_transient", "description": "用瞬态安全包络库快速估计一次切换在干管、分干管（阀前 PVC 区）的最大最小压力；毫秒级，不覆盖阀后 PE 支管",
     "parameters": {"type": "object", "properties": {"open_before": {"type": "array", "items": {"type": "string"}},
                                                     "actions": {"type": "array"}}, "required": ["open_before", "actions"]}},
    {"name": "simulate_transient", "description": "全量特征线法计算切换或故障处置动作，返回全网最大最小压力和逐点越限",
     "parameters": {"type": "object", "properties": {"open_before": {"type": "array", "items": {"type": "string"}},
                                                     "actions": {"type": "array"}}, "required": ["open_before", "actions"]}},
    {"name": "check_constraints", "description": "对完整决策 JSON 做全部约束检查（结构、最短启闭时间、稳态、瞬态）",
     "parameters": {"type": "object", "properties": {"decision": {"type": "object"},
                                                     "open_before": {"type": "array", "items": {"type": "string"}}},
                    "required": ["decision"]}},
    {"name": "query_node", "description": "查询节点、管段或阀门的高程、管径、材质、承压等级、最短启闭时间",
     "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "run_optimizer", "description": "调用 NSGA-II 优化器给出当前工况下的轮灌分组方案（约 30 s）",
     "parameters": {"type": "object", "properties": {}}},
]


if __name__ == "__main__":
    ctx = ev.Context()
    G = net.rotation_groups()
    print("query_node V05L:", query_node("V05L"))
    acts = rule_switch(G[1], G[2], 0)
    t0 = time.time(); s = screen_transient(ctx, G[1], acts)
    print(f"screen_transient {1000 * (time.time() - t0):.0f} ms:", {k: s[k] for k in ("pass_screen", "est_p_max_pvc", "est_p_min_pvc")})
    print("simulate_transient:", simulate_transient(ctx, G[1], acts)["extremes"])
    for k in (0, 1, 2):
        acts, r, lv = safe_switch(ctx, G[k], G[k + 1])
        print(f"safe_switch 组 {k + 1}→{k + 2}: 档位 {lv}，可行 {r['feasible']}，{r['extremes']}，越限 {r['violations'][:2]}")
