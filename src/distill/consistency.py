"""故障处置决策的隔离—计划一致性检查（R8）：纯规则、毫秒级、不调 API、不跑仿真。

正式实验的 check_constraints 按场景真值的隔离范围做稳态计算，不看决策自己声明的隔离：决策关了 VSx 却仍把分干管 x 的支管
排进剩余计划，稳态照样算出这些支管有水、判为可行。这里补上决策内部的自洽检查，蒸馏时用来过滤（to_sft R8）和给教师反馈（distill_gen）。
  plan_in_isolated    剩余计划排入了隔离阀下游的支管（关 VSx → 分干管 x 的 12 条支管全部退出；关 Vxks → 该支管退出）
  removed_in_plan     声明退出（removed_valves）的支管又排进剩余计划
  removed_missing     隔离子树内未灌完的支管没有列进 removed_valves
  isolate_not_closed  isolate_valves 里的阀在 isolation_actions 里没有关阀动作
  isolate_reopened    isolation_actions 又把隔离阀或其下游支管打开
  burst_not_isolated  爆管段不在隔离子树内（支管段 PBxks 关 Vxks 或 VSx，分干管段 Pxk 关 VSx）
  done_rescheduled    已灌完的支管又排进剩余计划（稳态口径只对未灌完支管记缺水 / 超灌，重排不受罚）
  unknown_valve       隔离阀或计划里出现不存在的阀名
  isolate_stuck       把卡开 / 部分卡死的阀当隔离阀（关不上，隔离不成立）
  isolate_off_fault   隔离阀不在故障位置上（按场景真值对照：管段 Pxk 只能关 VSx，PBxks 关 Vxks 或 VSx，干管段 Mx 没有隔离阀；
                      卡死阀 Vxks 只能关它自己或 VSx）。DeepSeek 测试集里有把 P04 读成 4 号分干管、去关 VS4 的
  stuck_actuated      对卡死阀本身下动作（动作不生效）。schema 要求 isolation_actions ≥ 1 项，教师常这样凑数，只记录不判不一致

生成端（distill_gen --consistency on，默认）用 install() 在本进程内换掉下面几处，不改正式实验的文件和进程：
  tools.check_constraints   稳态改按决策声明的隔离算（aware_context：I 关闭、子树支管退出，多隔离的支管需水计入缺水），
                            并加 checks.consistency；教师调用工具和每轮校核反馈都看得到，不一致即 feasible = false，要求修改
  teacher.execute_tool      simulate_steady 可带 isolate_valves，按隔离后的管网算（原来只认场景真值，排除隔离支管反被判 valve_unassigned）
  teacher.SCHEMA            fault_response.isolation_actions 允许 0 项（泄漏、卡死常常不需要隔离，minItems 1 逼教师凑动作）
  teacher.hard_constraints  在第 7 条后加第 8 条一致性要求；tools.TOOL_SPECS 里 check_constraints、simulate_steady 的说明同步
  teacher.network_summary   补管段编号说明（Mx / Pxk / PBxks 与分干管号的关系；DeepSeek 测试集里有把 P04 读成 4 号分干管的）

python distill/consistency.py --src distill/raw     # 统计各轨迹最终方案和中间方案的一致性
"""
import argparse
import collections
import copy
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import network as net                                       # noqa: E402

BRANCH = [v for v, *_ in net.branches()]
SUBMAIN = [f"VS{x}" for x in range(10)]
VALID = set(BRANCH) | set(SUBMAIN)
HARD = ("plan_in_isolated", "removed_in_plan", "removed_missing", "isolate_not_closed", "isolate_reopened",
        "burst_not_isolated", "done_rescheduled", "unknown_valve", "isolate_stuck", "isolate_off_fault")
NAMING = "管段编号：干管段 Mx（干管第 x 段，末端 Jx）；分干管段 Pxk（第一位数字是分干管号，第二位是段号：P04 是 0 号分干管第 4 段、" \
         "末端 J04，不是 4 号分干管）；支管段 PBxk{L,R}（支管阀 Vxk{L,R} 之后的 PE 支管）。隔离：分干管段 Pxk 没有中间阀，只能关 VSx，" \
         "整条分干管 x 的 12 条支管退出；支管段 PBxks 关 Vxks（或 VSx）；干管段没有隔离阀。"
RULE = "故障处置的隔离与计划必须自洽：isolate_valves 下游的支管（关 VSx 即分干管 x 的全部 12 条）列入 removed_valves 并退出剩余计划，" \
       "已灌完的支管不再排入；卡开或部分卡死的阀关不上，不能作隔离阀；泄漏和卡死不一定需要隔离，isolation_actions 可以为空。"


def downstream(v):
    return {u for u in BRANCH if u[1] == v[2]} if v.startswith("VS") else {v} & set(BRANCH)


def subtree(valves):
    return set().union(*(downstream(v) for v in valves)) if valves else set()


def burst_cover(pipe):
    """能隔离爆管段 pipe 的阀：支管段 PBxks → Vxks 或 VSx；分干管段 Pxk → VSx。"""
    if pipe.startswith("PB"):
        return {f"V{pipe[2:]}", f"VS{pipe[2]}"}
    m = re.fullmatch(r"P(\d)\d", pipe)
    return {f"VS{m.group(1)}"} if m else set()


def fault_cover(fault):
    """场景真值下允许的隔离阀集合；None 表示不检查。"""
    if fault.get("type") in ("burst", "leak") and fault.get("pipe"):
        return burst_cover(fault["pipe"])
    v = fault.get("valve") or ""
    if fault.get("type") == "stuck" and v in VALID:
        return {v} if v.startswith("VS") else {v, f"VS{v[1]}"}
    return None


def plan_valves(plan):
    return [v for g in (plan or {}).get("groups", []) for v in g.get("valves", [])]


def check(decision, fault, done=()):
    """decision: fault_response 决策 JSON；fault: 场景故障（type, pipe, valve, mode …）；done: 已灌完支管。
    返回 dict(consistent, issues=[dict(code, valves)], isolated_subtree)。"""
    fr = decision.get("fault_response") or {}
    iso = list(fr.get("isolate_valves") or [])
    acts = list(fr.get("isolation_actions") or [])
    removed = set(fr.get("removed_valves") or [])
    plan = plan_valves(fr.get("remaining_plan"))
    done = set(done)
    issues = []

    def add(code, vs):
        if vs:
            issues.append(dict(code=code, valves=sorted(vs)))

    add("unknown_valve", (set(iso) | set(plan) | {a.get("valve") for a in acts}) - VALID)
    sub = subtree([v for v in iso if v in VALID])
    add("plan_in_isolated", sub & set(plan))
    add("removed_in_plan", removed & set(plan))
    add("removed_missing", sub - done - removed)
    closed = {a.get("valve") for a in acts if a.get("op") == "close"}
    add("isolate_not_closed", {v for v in iso if v not in closed and not (v.startswith("V") and f"VS{v[1]}" in closed and f"VS{v[1]}" in iso)})
    add("isolate_reopened", {a.get("valve") for a in acts if a.get("op") == "open" and (a.get("valve") in iso or a.get("valve") in sub)})
    if fault.get("type") == "burst" and fault.get("pipe") and not burst_cover(fault["pipe"]) & set(iso):
        add("burst_not_isolated", {fault["pipe"]})
    add("done_rescheduled", done & set(plan))
    cover = fault_cover(fault)
    if cover is not None:
        add("isolate_off_fault", {v for v in iso if v in VALID} - cover)
    if fault.get("type") == "stuck":
        if fault.get("mode") != "stuck_closed":
            add("isolate_stuck", {fault["valve"]} & set(iso))
        add("stuck_actuated", {a.get("valve") for a in acts if a.get("valve") == fault.get("valve")})
    return dict(consistent=not any(i["code"] in HARD for i in issues), issues=issues, isolated_subtree=sorted(sub))


def repair(decision, fault, done=(), mode="plan"):
    """规则修复不一致的决策（零 API），两种口径：
      mode="plan"       保留隔离，把隔离子树和已灌完支管移出剩余计划（空组删掉、组号重排），removed_valves 补齐子树；
      mode="isolation"  保留计划，撤掉隔离阀和对应关阀动作（泄漏、卡死可以不隔离；爆管不用这一口径）。
    两种口径都把卡开 / 部分卡死阀移出隔离阀和动作。修复方案仍要过一致性版 check_constraints 才能用（to_sft --repair）。"""
    d = copy.deepcopy(decision)
    fr = d["fault_response"]
    stuck = {fault.get("valve")} if fault.get("type") == "stuck" and fault.get("mode") != "stuck_closed" else set()
    cover = fault_cover(fault)
    raw = [v for v in fr.get("isolate_valves") or [] if v in VALID]
    iso0 = [v for v in raw if v not in stuck and (cover is None or v in cover)]
    if mode == "plan" and fault.get("type") == "burst" and not iso0:          # 爆管没隔离对：补最小隔离（支管段关 Vxks，分干管段关 VSx）
        p = fault["pipe"]
        iso0 = [f"V{p[2:]}" if p.startswith("PB") else f"VS{p[1]}"]
    dropped = set(raw) - set(iso0)
    fr["isolate_valves"] = iso0 if mode == "plan" else []
    fr["isolation_actions"] = [a for a in fr.get("isolation_actions") or [] if a.get("valve") not in stuck | dropped
                               and not (mode == "isolation" and a.get("op") == "close" and a.get("valve") in iso0)]
    have = {a.get("valve") for a in fr["isolation_actions"] if a.get("op") == "close"}
    fr["isolation_actions"] += [dict(valve=v, op="close", t_start_s=0.0, template="close_two_stage",
                                     params=dict(duration_s=20 if v.startswith("VS") else 10, break_time_s=5, break_opening=0.3))
                                for v in fr["isolate_valves"] if v not in have]
    drop = subtree(fr["isolate_valves"]) | set(done)
    plan = fr["remaining_plan"]
    by_id = {g["group_id"]: g for g in plan["groups"]}
    groups = []
    for gid in plan.get("order") or list(by_id):
        vs = [v for v in by_id[gid]["valves"] if v not in drop]
        if vs:
            groups.append(dict(by_id[gid], group_id=len(groups) + 1, valves=vs))
    fr["remaining_plan"] = dict(groups=groups, order=list(range(1, len(groups) + 1)))
    kept = {v for g in groups for v in g["valves"]}
    fr["removed_valves"] = sorted(((set(fr.get("removed_valves") or []) | subtree(fr["isolate_valves"])) - set(done)) - kept)
    fixed = sorted({i["code"] for i in check(decision, fault, done)["issues"] if i["code"] in HARD})
    how = (f"隔离阀取 {fr['isolate_valves']}，退出支管 {len(fr['removed_valves'])} 条移出剩余计划" if mode == "plan" else
           f"{ {'leak': '泄漏', 'stuck': '卡死'}.get(fault.get('type'), '') }不必隔离，撤掉对 {sorted(set(raw)) or '卡死阀'} 的隔离，剩余计划照常")
    d["reasoning"] = (d.get("reasoning") or "")[:1250] + f" 一致性自检发现 {'、'.join(fixed)}：{how}，共 {len(groups)} 组。"
    return d


def aware_context(ctx, isolate_valves):
    """按决策声明的隔离改写稳态边界：I 里能关的阀关闭，子树支管退出可用集合。
    这样计划排入子树支管就是 isolated_valve_assigned，按声明不排也不再被判 valve_unassigned。
    返回 (ctx', 自愿退出的支管)：场景真值之外多隔离的未灌支管，它们的需水要记进缺水量，否则多隔离反而 C* 更高。"""
    fault = getattr(ctx, "fault", {}) or {}
    stuck = {fault.get("valve")} if fault.get("type") == "stuck" and fault.get("mode") != "stuck_closed" else set()
    iso = [v for v in isolate_valves or [] if v in VALID and v not in stuck]
    if not iso:
        return ctx, set()
    sub = subtree(iso)
    c = copy.copy(ctx)
    c.closed_links, c.removed = set(ctx.closed_links) | set(iso), set(ctx.removed) | sub
    c.valves = [v for v in ctx.valves if v not in sub]
    return c, set(ctx.valves) & sub


def steady_aware(ctx, isolate_valves, plan):
    """simulate_steady 的隔离感知版：稳态按声明的隔离算，自愿退出支管的需水计入缺水，C* 重算。"""
    import scoring
    import tools
    actx, extra = aware_context(ctx, isolate_valves)
    s = tools.simulate_steady(actx, plan)
    if extra:
        s["objectives"] = dict(s["objectives"], deficit_m3=round(s["objectives"]["deficit_m3"] + sum(ctx.D[v] for v in extra), 1))
        s["score"] = scoring.closeness(s["objectives"])
    return s


_orig = {}


def check_constraints(ctx, decision, open_before=None, events=(), skip_transient=False):
    """tools.check_constraints 的一致性版本：瞬态仍从隔离前状态算（原函数、原 ctx），稳态换成 steady_aware，再加 checks.consistency。"""
    out = _orig["check"](ctx, decision, open_before, events, skip_transient)
    if decision.get("decision_type") != "fault_response" or not hasattr(ctx, "fault"):
        return out
    fr = decision["fault_response"]
    s = steady_aware(ctx, fr.get("isolate_valves"), fr["remaining_plan"])
    out["checks"]["steady"] = dict(feasible=s["feasible"], violations=s["violations"], objectives=s["objectives"])
    c = check(decision, ctx.fault, ctx.done)
    out["checks"]["consistency"] = dict(feasible=c["consistent"], issues=c["issues"])
    out["feasible"] = all(v["feasible"] for v in out["checks"].values())
    return out


def execute_tool(name, args, ctx, sc, state):
    if name == "simulate_steady" and hasattr(ctx, "fault") and args.get("isolate_valves"):
        return steady_aware(ctx, args["isolate_valves"], args["rotation_plan"])
    return _orig["exec"](name, args, ctx, sc, state)


def install():
    import teacher
    import tools
    if _orig:
        return
    _orig.update(check=tools.check_constraints, make_state=teacher.make_state, hard=teacher.hard_constraints, exec=teacher.execute_tool,
                 net=teacher.network_summary)

    def make_state(task, sc, rng):
        ctx, st = _orig["make_state"](task, sc, rng)
        if task == "fault_response":
            ctx.fault, ctx.done = sc["faults"][0], list(st.get("done", []))
        return ctx, st

    teacher.make_state = make_state
    teacher.execute_tool = execute_tool
    teacher.network_summary = lambda *a, **k: _orig["net"](*a, **k).replace("\n观测：", f"\n{NAMING}\n观测：", 1)
    tools.check_constraints = check_constraints
    teacher.hard_constraints = lambda: _orig["hard"]().replace("\n目标：", f"\n8. {RULE}\n目标：", 1)
    p = teacher.SCHEMA["properties"]
    p["fault_response"]["properties"]["isolation_actions"] = dict(p["switch"]["properties"]["actions"], minItems=0)
    for t in tools.TOOL_SPECS:
        if t["name"] == "check_constraints" and "一致性" not in t["description"]:
            t["description"] += "；故障处置另查隔离与计划的一致性，稳态按决策声明的隔离计算"
        if t["name"] == "simulate_steady" and "isolate_valves" not in t["parameters"]["properties"]:
            t["description"] += "；故障处置时可传 isolate_valves，按隔离后的管网计算，多隔离的支管需水计入缺水"
            t["parameters"]["properties"]["isolate_valves"] = {"type": "array", "items": {"type": "string"}}


def candidates(r):
    """轨迹里教师给出的全部候选方案：工具参数里的完整决策（check_constraints）和每轮最终 JSON。"""
    import teacher
    out = []
    for m in r["messages"]:
        if m["role"] != "assistant":
            continue
        for c in m.get("tool_calls") or []:
            if c["function"]["name"] == "check_constraints":
                try:
                    d = json.loads(c["function"]["arguments"] or "{}").get("decision")
                except json.JSONDecodeError:
                    d = None
                if isinstance(d, dict) and d.get("decision_type") == "fault_response":
                    out.append(("tool", d))
        if not m.get("tool_calls"):
            try:
                out.append(("final", teacher.parse_decision(m.get("content") or "")))
            except Exception:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(HERE / "raw"))
    a = ap.parse_args()
    R = [json.load(open(p, encoding="utf-8")) for p in sorted(pathlib.Path(a.src).glob("fault_response/*.json"))]
    R = [r for r in R if r["status"] == "ok"]
    by = collections.defaultdict(collections.Counter)
    for r in R:
        f, done, o = r["scenario"]["fault"], r["state"]["done"], r["outcome"]
        t = f["type"]
        if o.get("decision"):
            c = check(o["decision"], f, done)
            by[t]["final"] += 1
            by[t]["final_bad"] += not c["consistent"]
            by[t]["final_bad_but_feasible"] += (not c["consistent"]) and bool(o.get("final_feasible")) and o["feasible_before_fallback"]
            for i in c["issues"]:
                by[t]["final:" + i["code"]] += 1
        if o.get("final_decision"):
            by[t]["delivered_bad"] += not check(o["final_decision"], f, done)["consistent"]
        for kind, d in candidates(r):
            c = check(d, f, done)
            by[t][f"cand_{kind}"] += 1
            by[t][f"cand_{kind}_bad"] += not c["consistent"]
    for t, c in sorted(by.items()):
        print(t, dict(c))


if __name__ == "__main__":
    main()
