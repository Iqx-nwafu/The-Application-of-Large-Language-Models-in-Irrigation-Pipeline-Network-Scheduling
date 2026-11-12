"""向量化特征线法（MOC）瞬态内核，自研平台接入前的快速替代，用 TSNet 交叉验证（validate_moc.py）。

格式：所有管道的计算点拼成一维数组，一步内整体更新；节点、阀门、水库边界也全部向量化。
  内点     H = (CP + CM)/2，Q = (CP − CM)/(2B)，B = a/(gA)，R = 每段稳态摩阻（准稳态，按初始稳态标定）
  节点     Σ (Cx − H)/B = q_ext；q_ext = C·sqrt(H − z)（滴头、泄漏、爆管）解一元二次；负压时出流取 0
  阀门     两端各接一根管：Q = c(t)·sign(ΔH)·sqrt|ΔH|，c = Cv_rel(s)·A_v·sqrt(2g/K)
  水库     定水头
时间步 dt 默认 0.02 s（最短管 6 m / 300 m/s），各管段数取整后按 TSNet 的做法微调波速。
支管末端 E 节点通过滴灌带与大气相通：压力降到 0 以下时进气，按 p = 0 处理（进气边界，不跟踪气囊体积）。
其他位置负压低于汽化压力（约 −10 m）时不做液柱分离计算，只在结果里标记 below_vapor。

初始稳态来自 hydraulics.SteadyModel，阀门 K、滴头规律、泄漏和部分开度与稳态完全一致。

用法
  sim = TransientModel()
  out = sim.run(open_valves=G1, actions=[...], tf=40)       # actions 与 decision_output.schema 的 switch.actions 相同
  out["p_max"], out["p_min"]（节点）, out["pipe_p_max"], out["series"]
"""
import pathlib
import numpy as np
import wntr
import network as net
from hydraulics import SteadyModel, cv_rel, C_EMIT, G, P_B_CAP

HERE = pathlib.Path(__file__).parent
A_PVC, A_PE = 300.0, 100.0
P_VAPOR = -10.0


def is_pe(name):
    return name.startswith(("PC", "PB"))


# ---------------- 启闭模板（valve_templates.json） ----------------
def opening_curve(action, t, s0, s1=1.0):
    """action: {op, t_start_s, template, params}；返回 t 上的开度序列。
    关阀从当前开度 s0 关到 0；开阀从 0 开到目标开度 s1（切换后稳态自动节流算出的开度）；动作前保持 s0。"""
    t0, p = action["t_start_s"], action.get("params", {})
    d = 1.0 if action["template"] == "instant" else float(p["duration_s"])
    u = np.clip((t - t0) / d, 0, 1)
    tau = np.clip(t - t0, 0, d)
    if action["template"] in ("close_two_stage", "open_two_stage"):
        tb, sb = float(p["break_time_s"]), float(p["break_opening"])
        if action["op"] == "close":
            s = np.where(tau < tb, 1 - (1 - sb) * tau / tb, sb * (1 - (tau - tb) / (d - tb)))
        else:
            s = np.where(tau < tb, sb * tau / tb, sb + (1 - sb) * (tau - tb) / (d - tb))
    else:
        s = 1 - u if action["op"] == "close" else u
    s = s * (s0 if action["op"] == "close" else s1)
    s = np.where(t < t0, s0, s)
    return np.clip(s, 0, 1)

class TransientModel:
    def __init__(self, inp=HERE / "benchmark71.inp", dt=0.02, rough_mult=1.0, wavespeed_mult=1.0):
        self.steady = SteadyModel(inp, rough_mult)
        wn = wntr.network.WaterNetworkModel(str(inp))
        self.wn = wn
        self.res = wn.reservoir_name_list[0]
        self.node_names = wn.junction_name_list + [self.res]
        self.nidx = {n: i for i, n in enumerate(self.node_names)}
        self.z = np.array([wn.get_node(n).elevation for n in wn.junction_name_list] + [0.0])
        self.H_res = wn.get_node(self.res).base_head
        self.dt = dt
        # 管道离散
        self.pipes = wn.pipe_name_list
        a, n_seg, off = [], [], [0]
        for p in self.pipes:
            o = wn.get_link(p)
            a0 = (A_PE if is_pe(p) else A_PVC) * wavespeed_mult
            n = max(1, int(round(o.length / (a0 * dt))))
            n_seg.append(n); a.append(o.length / (n * dt)); off.append(off[-1] + n + 1)
        self.a, self.n_seg = np.array(a), np.array(n_seg)
        self.off = np.array(off[:-1])
        self.npts = off[-1]
        self.pD = np.array([wn.get_link(p).diameter for p in self.pipes])
        self.pA = np.pi * self.pD ** 2 / 4
        self.pL = np.array([wn.get_link(p).length for p in self.pipes])
        self.B_pipe = self.a / (G * self.pA)
        pid = np.repeat(np.arange(len(self.pipes)), self.n_seg + 1)
        self.pt_pipe = pid
        self.pt_B = self.B_pipe[pid]
        loc = np.arange(self.npts) - self.off[pid]
        self.pt_frac = loc / self.n_seg[pid]
        self.interior = np.where((loc > 0) & (loc < self.n_seg[pid]))[0]
        self.up_end = self.off                          # 管道起点（C− 特征线）
        self.dn_end = self.off + self.n_seg             # 管道终点（C+ 特征线）
        s_node = np.array([self.nidx[wn.get_link(p).start_node_name] for p in self.pipes])
        e_node = np.array([self.nidx[wn.get_link(p).end_node_name] for p in self.pipes])
        self.s_node, self.e_node = s_node, e_node
        self.pt_z = self.z[s_node[pid]] + (self.z[e_node[pid]] - self.z[s_node[pid]]) * self.pt_frac
        # 管端表：每个管端一行（点号、所属节点、+1 终点 / −1 起点、B）
        self.end_pt = np.r_[self.dn_end, self.up_end]
        self.end_node = np.r_[e_node, s_node]
        self.end_sign = np.r_[np.ones(len(self.pipes)), -np.ones(len(self.pipes))]
        self.end_B = np.r_[self.B_pipe, self.B_pipe]
        # 阀门
        self.valves = wn.valve_name_list
        self.v_u = np.array([self.nidx[wn.get_link(v).start_node_name] for v in self.valves])
        self.v_d = np.array([self.nidx[wn.get_link(v).end_node_name] for v in self.valves])
        self.v_A = np.array([np.pi * wn.get_link(v).diameter ** 2 / 4 for v in self.valves])
        self.v_K = np.array([float(wn.get_link(v).initial_setting) for v in self.valves])
        self.vidx = {v: i for i, v in enumerate(self.valves)}
        valve_nodes = set(self.v_u) | set(self.v_d)
        for n in valve_nodes:
            assert (self.end_node == n).sum() == 1, f"阀门端节点 {self.node_names[n]} 必须恰好接一根管"
        self.v_u_end = np.array([np.where(self.end_node == n)[0][0] for n in self.v_u])
        self.v_d_end = np.array([np.where(self.end_node == n)[0][0] for n in self.v_d])
        ires = self.nidx[self.res]
        gen = [i for i in range(len(self.node_names)) if i not in valve_nodes and i != ires]
        self.gen_nodes = np.array(gen)
        self.end_is_gen = np.isin(self.end_node, self.gen_nodes)
        self.end_is_res = self.end_node == ires
        self.e_nodes = np.array([self.nidx[e] for v, _, e in net.branches() if v in self.vidx], int)
        self.branch_valves = [v for v, *_ in net.branches() if v in self.vidx]

    # ---------------- 初始稳态 ----------------
    def initial(self, open_valves, emitters, openings, closed_links, c_override, p_cap):
        st = self.steady
        r = st.solve([set(open_valves)], extra_emitters=[emitters], openings=openings,
                     closed_links=closed_links, c_override=[c_override] if c_override else None, p_cap=p_cap)
        alive = r["alive"][0]
        Hn = np.zeros(len(self.node_names))
        for n, i in st.idx.items():
            Hn[self.nidx[n]] = r["H"][0, i] if alive[i] else self.z[self.nidx[n]]
        Hn[self.nidx[self.res]] = self.H_res
        # 管道流量按 INP 方向
        Qp = np.zeros(len(self.pipes))
        for j, p in enumerate(self.pipes):
            i = st.lidx[p]
            Qp[j] = r["Q"][0, i] / 3600 * (1 if st.nodes[i] == self.node_names[self.e_node[j]] else -1)
        Hs, He = Hn[self.s_node], Hn[self.e_node]
        R = np.where(np.abs(Qp) > 1e-7, (Hs - He) / self.n_seg / np.maximum(Qp * np.abs(Qp), 1e-14), 0.0)
        # 零流量管段用设计流量下的摩阻
        f0 = 0.02
        R = np.where(np.abs(Qp) > 1e-7, R, f0 * (self.pL / self.n_seg) / (2 * G * self.pD * self.pA ** 2))
        pid = self.pt_pipe
        H = Hs[pid] + (He - Hs)[pid] * self.pt_frac
        Q = Qp[pid].copy()
        Cn = np.zeros(len(self.node_names))                    # 节点出流系数（含关阀支管，开阀后自然出流）
        Cn[self.e_nodes] = C_EMIT
        for n, c in (c_override or {}).items():
            Cn[self.nidx[n]] = c
        for n, c in emitters.items():
            Cn[self.nidx[n]] += c
        return H, Q, R[pid], Cn, r

    def valve_c(self, s):
        return cv_rel(s) * self.v_A * np.sqrt(2 * G / self.v_K)

    # ---------------- 主循环 ----------------
    def run(self, open_valves, actions=(), tf=40.0, emitters=None, openings=None, closed_links=(),
            events=(), stuck=(), c_override=None, series_nodes=(), p_cap=P_B_CAP):
        """open_valves: 初始开启的支管阀；actions: 阀门动作列表；
        events: [(节点, 出流系数 C m3/s/m^0.5, 起始时刻 s, 历时 s)] 突发爆管；
        stuck: 卡死阀（动作被忽略，保持初始/卡死开度）；
        p_cap: 支管入口稳态上限，切换前后的稳态都按它自动节流，开阀开到切换后稳态的目标开度。"""
        emitters, openings, stuck = emitters or {}, dict(openings or {}), set(stuck)
        H, Q, R, Cn, r0 = self.initial(open_valves, emitters, openings, closed_links, c_override, p_cap)
        nt = int(round(tf / self.dt)) + 1
        t = np.arange(nt) * self.dt
        # 阀门开度表 (nt, nv)：初始开度来自切换前稳态（含自动节流）
        s0 = np.ones(len(self.valves))
        for v, s in zip(self.steady.valves, r0["opening"][0]):
            s0[self.vidx[v]] = s
        for v in closed_links:
            s0[self.vidx[v]] = 0.0
        for v, s in openings.items():
            s0[self.vidx[v]] = s
        # 切换后稳态：开阀的目标开度
        closing = {a["valve"] for a in actions if a["op"] == "close" and a["valve"] not in stuck}
        opening_v = {a["valve"] for a in actions if a["op"] == "open" and a["valve"] not in stuck}
        branch = set(self.branch_valves)
        after = (set(open_valves) - closing) | (opening_v & branch)
        closed_after = (set(closed_links) - opening_v) | (closing - branch)
        r1 = self.steady.solve([after], extra_emitters=[emitters], openings=openings, closed_links=sorted(closed_after),
                               c_override=[c_override] if c_override else None, p_cap=p_cap)
        s1 = dict(zip(self.steady.valves, r1["opening"][0]))
        S = np.tile(s0, (nt, 1))
        for act in actions:
            if act["valve"] in stuck:
                continue
            j = self.vidx[act["valve"]]
            S[:, j] = opening_curve(act, t, S[0, j], s1.get(act["valve"], 1.0))
        # 部分开度：稳态 K 已按开度换算，c 需要相对全开曲线计算
        Cv = cv_rel(S) * self.v_A * np.sqrt(2 * G / self.v_K)
        ev_node = np.array([self.nidx[e[0]] for e in events], int)
        ev = [(self.nidx[n], c, ts, max(tc, self.dt)) for n, c, ts, tc in events]

        z = self.z
        gen = self.gen_nodes
        nN = len(self.node_names)
        Bend = self.end_B
        pmax_pt, pmin_pt = H - self.pt_z, H - self.pt_z
        pn = H[self.end_pt] - z[self.end_node]
        node_pmax = np.full(nN, -np.inf); node_pmin = np.full(nN, np.inf)
        ser_idx = [self.nidx[n] for n in series_nodes]
        ser = np.zeros((nt, len(ser_idx)))
        Hn = np.zeros(nN)
        Hn[self.end_node] = H[self.end_pt]
        ser[0] = Hn[ser_idx] - z[ser_idx]
        ires = self.nidx[self.res]
        prev = np.r_[self.dn_end - 1, self.up_end + 1]
        Ssum = np.bincount(self.end_node, weights=1 / Bend, minlength=nN)      # 管端特征线来源点
        pt_B, interior = self.pt_B, self.interior
        for k in range(1, nt):
            Hp, Qp = H, Q
            # 内点
            a, b = interior - 1, interior + 1
            CP = Hp[a] + pt_B[a] * Qp[a] - R[a] * Qp[a] * np.abs(Qp[a])
            CM = Hp[b] - pt_B[b] * Qp[b] + R[b] * Qp[b] * np.abs(Qp[b])
            H = Hp.copy(); Q = Qp.copy()
            H[interior] = 0.5 * (CP + CM)
            Q[interior] = (CP - CM) / (2 * pt_B[interior])
            # 管端特征值 Cx：终点用 C+，起点用 C−
            pr = prev
            Rp = R[pr]
            Cx = np.where(self.end_sign > 0,
                          Hp[pr] + pt_B[pr] * Qp[pr] - Rp * Qp[pr] * np.abs(Qp[pr]),
                          Hp[pr] - pt_B[pr] * Qp[pr] + Rp * Qp[pr] * np.abs(Qp[pr]))
            # 一般节点
            Y = np.bincount(self.end_node, weights=Cx / Bend, minlength=nN)
            Ce = Cn.copy()
            for i, c, ts, tc in ev:
                Ce[i] += c * np.clip((t[k] - ts) / tc, 0, 1)
            Sg, Yg, zg, cg = Ssum[gen], Y[gen], z[gen], Ce[gen]
            disc = Yg - Sg * zg
            y = np.where(disc > 0, (-cg + np.sqrt(cg ** 2 + 4 * Sg * np.maximum(disc, 0))) / (2 * Sg), 0.0)
            Hn[gen] = np.where(disc > 0, zg + y ** 2, Yg / Sg)
            Hn[self.e_nodes] = np.maximum(Hn[self.e_nodes], z[self.e_nodes])      # 滴灌带进气
            Hn[ires] = self.H_res
            # 阀门
            c = Cv[k]
            Cu, Cd = Cx[self.v_u_end], Cx[self.v_d_end]
            Bs = Bend[self.v_u_end] + Bend[self.v_d_end]
            dlt = Cu - Cd
            c2 = c ** 2
            qv = np.sign(dlt) * (-c2 * Bs + np.sqrt(c2 ** 2 * Bs ** 2 + 4 * c2 * np.abs(dlt))) / 2
            Hn[self.v_u] = Cu - Bend[self.v_u_end] * qv
            Hn[self.v_d] = Cd + Bend[self.v_d_end] * qv
            # 回写管端
            He = Hn[self.end_node]
            Qe = self.end_sign * (Cx - He) / Bend
            H[self.end_pt] = He
            Q[self.end_pt] = Qe
            p = H - self.pt_z
            np.maximum(pmax_pt, p, out=pmax_pt); np.minimum(pmin_pt, p, out=pmin_pt)
            pn = Hn - z
            np.maximum(node_pmax, pn, out=node_pmax); np.minimum(node_pmin, pn, out=node_pmin)
            if ser_idx:
                ser[k] = pn[ser_idx]
        pipe_pmax = np.maximum.reduceat(pmax_pt, self.off)
        pipe_pmin = np.minimum.reduceat(pmin_pt, self.off)
        return dict(t=t, p_max=dict(zip(self.node_names, node_pmax)), p_min=dict(zip(self.node_names, node_pmin)),
                    pipe_p_max=dict(zip(self.pipes, pipe_pmax)), pipe_p_min=dict(zip(self.pipes, pipe_pmin)),
                    series=dict(zip(series_nodes, ser.T)), below_vapor=bool(pmin_pt.min() < P_VAPOR), steady=r0,
                    opening_before=dict(zip(self.steady.valves, r0["opening"][0])), opening_after=s1)


if __name__ == "__main__":
    import time
    sim = TransientModel()
    G1, G2 = net.rotation_groups()[0], net.rotation_groups()[1]
    acts = [dict(valve=v, op="close", t_start_s=2.0, template="close_linear", params=dict(duration_s=2.0)) for v in G1]
    acts += [dict(valve=v, op="open", t_start_s=2.0, template="open_linear", params=dict(duration_s=2.0)) for v in G2]
    t0 = time.time()
    out = sim.run(G1, acts, tf=60, series_nodes=["J0", "J02", "J05"])
    print(f"dt {sim.dt} s，计算点 {sim.npts}，60 s 模拟 CPU {time.time() - t0:.2f} s")
    sub = [f"J0{k}" for k in range(1, 7)]
    print("分干管 J0 最大 %.1f m，最小 %.1f m；干管最大 %.1f m" % (
        max(out["p_max"][n] for n in sub), min(out["p_min"][n] for n in sub), max(out["p_max"][f"J{i}"] for i in range(10))))
