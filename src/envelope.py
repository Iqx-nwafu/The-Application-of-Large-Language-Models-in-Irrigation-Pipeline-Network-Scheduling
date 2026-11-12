"""A6 瞬态安全包络库：离线单阀表 + 在线叠加筛查 + 全量仿真检验漏判率。

离线：对每个阀门 × 每个模板参数档 × 每档背景流量（同一分干管上另开 b 条支管），用 moc.py 算单阀动作，
      记录全部节点和管段相对稳态的超调：
        up   = max_t p − max(p_前, p_后)      （高于前后两个稳态的瞬态升压）
        down = min(p_前, p_后) − min_t p      （低于前后两个稳态的瞬态降压）
模板参数只允许取 GRID 里的离散档（valve_templates.json 的 allowed），模型输出的参数先投影到最近的保守档，
      所以单阀查表是精确的，误差只来自多阀叠加。
在线：多阀切换的估计
        p_max ≈ max(p_前, p_后) + 叠加(up) + margin
        p_min ≈ min(p_前, p_后) − 叠加(down) − margin
      叠加：起止窗口相隔不到 W 秒的动作归为一簇，簇内求和，簇间取最大。
      升压项按（实际稳态压力 / 建表稳态压力）放大，下限 1、上限 3。
      查表：同模板中历时不长于请求值的最慢一档（更快的动作水锤更大，偏保守）；
            背景 = 动作开始时同一分干管上仍开着的支管（已开始关的算关），向下取档。
检验：随机切换方案做全量计算，比较“筛查通过但实际越限”的漏判率，按校准集定 margin。

python envelope.py build      # 生成 envelope_lib.npz（多进程）
python envelope.py validate   # 漏判率检验 → envelope_validation.csv
"""
import json
import pathlib
import sys
import time
import numpy as np
from multiprocessing import Pool
import network as net
import transient_tools as tt

HERE = pathlib.Path(__file__).parent
LIB = HERE / "envelope_lib.npz"
LEVELS = [0, 1, 3]                         # 同一分干管上其他开启支管数（分干管阀为开启支管数 − 1）
W_CLUSTER = 10.0                           # s，叠加窗口
GRID = {
    "branch": {
        "close": [("close_linear", dict(duration_s=d)) for d in (5, 10, 20, 40)] +
                 [("close_two_stage", dict(duration_s=d, break_time_s=b, break_opening=s))
                  for d, b, s in ((8, 2, 0.3), (15, 3, 0.3), (20, 5, 0.3), (30, 5, 0.2), (40, 10, 0.2))] +
                 [("instant", dict(duration_s=1))],
        "open": [("open_linear", dict(duration_s=d)) for d in (5, 10, 20, 40)] +
                [("open_two_stage", dict(duration_s=d, break_time_s=b, break_opening=s))
                 for d, b, s in ((10, 5, 0.2), (15, 8, 0.2), (30, 10, 0.3))] +
                [("instant", dict(duration_s=1))],
    },
    "sub": {
        "close": [("close_linear", dict(duration_s=d)) for d in (10, 20, 40, 60)] + [("instant", dict(duration_s=1))],
        "open": [("open_linear", dict(duration_s=d)) for d in (10, 20, 40, 60)] + [("instant", dict(duration_s=1))],
    },
}


def neighbors(v, b):
    """同一分干管上离阀门 v 最近的 b 条支管。"""
    x, k = (v[2], 0) if v.startswith("VS") else (v[1], int(v[2]))
    cand = [u for u, *_ in net.branches() if u[1] == x and u != v]
    return sorted(cand, key=lambda u: (abs(int(u[2]) - k), u))[:b]


def entries():
    out = []
    for v in [u for u, *_ in net.branches()] + [f"VS{x}" for x in range(10)]:
        cls = "sub" if v.startswith("VS") else "branch"
        for op, temps in GRID[cls].items():
            for ti, (tmpl, p) in enumerate(temps):
                for b in LEVELS:
                    out.append(dict(valve=v, cls=cls, op=op, ti=ti, template=tmpl, params=p, level=b))
    return out


_sim, _st = None, None


def _init():
    global _sim
    from moc import TransientModel
    _sim = TransientModel()


def locations(sim):
    return sim.node_names + sim.pipes


def _pressures(out, sim):
    pmax = np.array([out["p_max"][n] for n in sim.node_names] + [out["pipe_p_max"][p] for p in sim.pipes])
    pmin = np.array([out["p_min"][n] for n in sim.node_names] + [out["pipe_p_min"][p] for p in sim.pipes])
    return pmax, pmin


def steady_p(sim, open_valves, closed_links=(), emitters=None):
    """前后稳态压力，按 locations 顺序（管段取两端较大/较小值）。死区（关阀下游）取 0。
    emitters：额外孔口出流（难瞬态子集的爆管出流）；None 与原口径相同。"""
    st = sim.steady
    r = st.solve([set(open_valves)], closed_links=closed_links, extra_emitters=[emitters] if emitters else None)
    p = np.zeros(len(sim.node_names))
    for n, i in st.idx.items():
        p[sim.nidx[n]] = r["p"][0, i] if r["alive"][0, i] else 0.0
    p[sim.nidx[sim.res]] = sim.H_res
    hi = np.r_[p, np.maximum(p[sim.s_node], p[sim.e_node])]
    lo = np.r_[p, np.minimum(p[sim.s_node], p[sim.e_node])]
    return hi, lo


def _run_entry(e):
    v, sim = e["valve"], _sim
    bg = neighbors(v, e["level"] + (e["cls"] == "sub"))        # 分干管阀：背景 = level + 1 条支管
    if e["cls"] == "branch":
        before = set(bg) | ({v} if e["op"] == "close" else set())
        after = set(bg) | ({v} if e["op"] == "open" else set())
        closed_b = closed_a = ()
    else:
        before = after = set(bg)
        closed_b = () if e["op"] == "close" else (v,)
        closed_a = (v,) if e["op"] == "close" else ()
    act = dict(valve=v, op=e["op"], t_start_s=1.0, template=e["template"], params=e["params"])
    out = sim.run(sorted(before), [act], tf=1.0 + e["params"]["duration_s"] + tt.SETTLE_S, closed_links=closed_b)
    pmax, pmin = _pressures(out, sim)
    hb, lb = steady_p(sim, before, closed_b)
    ha, la = steady_p(sim, after, closed_a)
    up = np.maximum(pmax - np.maximum(hb, ha), 0)
    down = np.maximum(np.minimum(lb, la) - pmin, 0)
    return up.astype(np.float32), down.astype(np.float32), np.maximum(hb, ha).astype(np.float32)


def build(workers=22):
    E = entries()
    t0 = time.time()
    with Pool(workers, initializer=_init) as pool:
        res = pool.map(_run_entry, E, chunksize=8)
    _init()
    np.savez_compressed(LIB, up=np.array([r[0] for r in res]), down=np.array([r[1] for r in res]),
                        p_ref=np.array([r[2] for r in res]),
                        meta=json.dumps(dict(entries=E, locations=locations(_sim), levels=LEVELS, grid=GRID)))
    print(f"包络库 {len(E)} 条，{time.time() - t0:.0f} s（{workers} 进程）→ {LIB.name}")


class Envelope:
    def __init__(self, path=LIB, margin=0.0):
        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        self.up, self.down, self.p_ref = z["up"], z["down"], z["p_ref"]
        self.locs = meta["locations"]
        self.key = {(e["valve"], e["op"], e["template"], e["ti"], e["level"]): i for i, e in enumerate(meta["entries"])}
        self.grid = meta["grid"]
        self.margin = margin
        self.pe = np.array([tt.is_pe(n) for n in self.locs])
        self.res = [i for i, n in enumerate(self.locs) if n == "Hub"]
        li = {n: i for i, n in enumerate(self.locs)}
        self.branch_locs = {v: (li[f"C{v[1:]}"], [li[f"B{v[1:]}"], li[f"E{v[1:]}"], li[f"PB{v[1:]}"]])
                            for v, *_ in net.branches()}

    def lookup(self, a, open_now):
        v = a["valve"]
        cls = "sub" if v.startswith("VS") else "branch"
        b = len([u for u in open_now if u != v and u[1] == (v[2] if cls == "sub" else v[1]) and not u.startswith("VS")])
        level = max(l for l in LEVELS if l <= max(b - (cls == "sub"), 0))   # 背景向下取档（背景越少水锤越大）
        d = 1.0 if a["template"] == "instant" else a["params"]["duration_s"]
        temps = self.grid[cls][a["op"]]
        cands = [i for i, (t, p) in enumerate(temps) if t == a["template"] and p["duration_s"] <= d]
        if cands:
            ti = max(cands, key=lambda i: temps[i][1]["duration_s"])
            tmpl = a["template"]
        else:                                   # 比表里最快一档还快：按 instant
            ti = len(temps) - 1; tmpl = "instant"
        return self.key[(v, a["op"], tmpl, ti, level)]

    def screen(self, p_before, p_after, actions, open_before):
        """p_before/p_after: (hi, lo) 两个稳态（envelope.steady_p）。返回 (p_max 估计, p_min 估计)。"""
        acts = sorted(actions, key=lambda a: a["t_start_s"])
        idx, win = [], []
        for a in acts:
            # 动作开始时的背景：已开始关的阀算关闭，开阀动作已完成的才算开启
            t = a["t_start_s"]
            now = {v for v in open_before if not any(b["valve"] == v and b["op"] == "close" and b["t_start_s"] <= t for b in acts)}
            now |= {b["valve"] for b in acts if b["op"] == "open" and tt.action_end(b) <= t}
            idx.append(self.lookup(a, now))
            win.append((a["t_start_s"], tt.action_end(a)))
        # 时间窗聚簇
        clusters, cur, end = [], [], -1e9
        for i, (s, e) in enumerate(win):
            if cur and s > end + W_CLUSTER:
                clusters.append(cur); cur = []
            cur.append(i); end = max(end, e) if cur[:-1] else e
        clusters.append(cur)
        # 升压按实际稳态压力与建表时稳态压力之比放大（开阀充水的超调随驱动水头增大）
        p_hi = np.maximum(p_before[0], p_after[0])
        scale = np.clip(p_hi / np.maximum(self.p_ref[idx], 1.0), 1.0, 3.0)
        up_i = self.up[idx] * scale
        up = np.max([up_i[c].sum(axis=0) for c in clusters], axis=0)
        down = np.max([self.down[[idx[i] for i in c]].sum(axis=0) for c in clusters], axis=0)
        pmax = np.maximum(p_before[0], p_after[0]) + up + self.margin
        pmin = np.minimum(p_before[1], p_after[1]) - down - self.margin
        return pmax, pmin

    def violations(self, pmax, pmin):
        lim = np.where(self.pe, tt.PN_PE, tt.PN_PVC) * tt.ALPHA
        vmax = pmax - lim
        vmin = np.where(self.pe, -np.inf, tt.H_VAC - pmin)       # 阀后 PE 部分不校核负压（见 transient_tools）
        vmax[self.res] = vmin[self.res] = -np.inf
        return vmax, vmin


# ---------------- 漏判率检验 ----------------
def random_switch(rng, groups):
    """随机切换：相邻两组，动作从模板档里随机取，起始时刻随机错开（30% 的样本只用最快几档，保证有越限）。"""
    k = rng.integers(len(groups) - 1)
    g1, g2 = groups[k], groups[k + 1]
    fast = rng.uniform() < 0.3
    acts, t = [], 0.0
    for op, vs in (("open", g2), ("close", g1)):
        temps = [x for x in GRID["branch"][op] if x[0] != "instant"]
        if fast:
            temps = [x for x in temps if x[1]["duration_s"] <= 10]
        for v in vs:
            tmpl, p = temps[rng.integers(len(temps))]
            acts.append(dict(valve=v, op=op, t_start_s=t, template=tmpl, params=dict(p)))
            t += float(rng.integers(0, 4))
        t += float(rng.integers(0, 8))
    return g1, g2, acts


def _val_case(args):
    i, g1, g2, acts = args
    sim = _sim
    out = sim.run(sorted(g1), acts, tf=max(tt.action_end(a) for a in acts) + tt.SETTLE_S)
    pmax, pmin = _pressures(out, sim)
    return i, pmax.astype(np.float32), pmin.astype(np.float32), steady_p(sim, g1), steady_p(sim, g2)


def validate(n=1200, workers=22, seed=7):
    import pandas as pd
    rng = np.random.default_rng(seed)
    plans = [net.rotation_groups()]
    opt = HERE / "opt_nominal.json"
    if opt.exists():
        plans.append([g["valves"] for g in json.load(open(opt, encoding="utf-8"))["best"]["per_group"]])
    cases = [(i, *random_switch(rng, plans[i % len(plans)])) for i in range(n)]
    t0 = time.time()
    with Pool(workers, initializer=_init) as pool:
        res = pool.map(_val_case, cases, chunksize=4)
    cpu = time.time() - t0
    env = Envelope(margin=0.0)
    rows = []
    for (i, g1, g2, acts), (_, pmax, pmin, sb, sa) in zip(cases, res):
        emax, emin = env.screen(sb, sa, acts, g1)
        tv_max, tv_min = env.violations(pmax.copy(), pmin.copy())
        ok = np.isfinite(pmax)
        ev_max, ev_min = env.violations(emax, emin)
        row = dict(case=i, plan=i % len(plans), n_act=len(acts))
        for zone, z in (("pvc", ok & ~env.pe), ("pe", ok & env.pe)):
            row.update({f"{zone}_true_max_excess": float(np.max(tv_max[z])), f"{zone}_true_min_excess": float(np.max(tv_min[z])),
                        # 估计误差（估计 − 实际），负值就是低估
                        f"{zone}_err_max": float(np.min((emax - pmax)[z])),
                        f"{zone}_est_max_excess": float(np.max(ev_max[z])), f"{zone}_est_min_excess": float(np.max(ev_min[z]))})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "envelope_validation.csv", index=False, float_format="%.3f")
    # 随机一半校准 margin（使校准集零漏判），另一半检验；阀前 PVC 区和阀后 PE 区分开统计
    cal_mask = np.random.default_rng(seed + 1).uniform(size=len(df)) < 0.5
    cal, test = df[cal_mask], df[~cal_mask]

    def rates(d, m, zone):
        true_v = (d[f"{zone}_true_max_excess"] > 0) | (d[f"{zone}_true_min_excess"] > 0)
        flag = (d[f"{zone}_est_max_excess"] + m > 0) | (d[f"{zone}_est_min_excess"] + m > 0)
        miss = int((true_v & ~flag).sum())
        return dict(n=len(d), n_true_violation=int(true_v.sum()), miss=miss, miss_rate=round(miss / max(true_v.sum(), 1), 4),
                    false_alarm=round(float((flag & ~true_v).sum() / max((~true_v).sum(), 1)), 4))

    print(f"全量计算 {n} 个切换，{cpu:.0f} s（{workers} 进程）")
    summary = {}
    for zone in ("pvc", "pe"):
        m_need = next(float(m) for m in np.arange(0, 40.01, 0.5) if rates(cal, m, zone)["miss"] == 0)
        summary[zone] = dict(margin_m=m_need, worst_underestimate_m=round(float(df[f"{zone}_err_max"].min()), 2),
                             cal=rates(cal, m_need, zone), test=rates(test, m_need, zone), test_margin0=rates(test, 0.0, zone))
        for p in sorted(df.plan.unique()):
            d = df[df.plan == p]
            summary[zone][f"true_violation_plan{p}"] = int(((d[f"{zone}_true_max_excess"] > 0) | (d[f"{zone}_true_min_excess"] > 0)).sum())
        print(zone, json.dumps(summary[zone], ensure_ascii=False))
    json.dump(dict(n=n, plans=["基线轮灌表", "NSGA-II 稳态最优"][:len(plans)], **summary),
              open(HERE / "envelope_margin.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    {"build": build, "validate": validate}[sys.argv[1]]()
