"""快速稳态求解器：树状管网直接递推，可对一批工况向量化求解。

支管末端 Exk 按滴头出流规律 q = q_d·(p/h_d)^x（q_d = 47 m3/h，h_d = 10 m，x = 0.5），
所以组内实际流量随压力变化；泄漏和爆管也按孔口 q = C·sqrt(p) 处理。
摩阻 Darcy-Weisbach + Swamee-Jain（与 EPANET 相同），阀门 TCV 局部损失 K·v²/2g（按阀门口径）。
阀门部分开度 s 时 K_eff = K / Cv_rel(s)²，Cv_rel = sqrt(相对 1/K)，取 valve_templates.json 的通用闸阀曲线。
支管入口（阀后 B 节点）稳态压力超过 P_B_CAP = 22.5 m（PE PN 25.5 m 留 3 m 瞬态裕量）的支管阀自动节流到该值，
开度由稳态算出放在结果 opening 里，不进决策空间（作者 2026-10-27）。

用法
  m = SteadyModel()
  r = m.solve([group1_valves, group2_valves, ...])     # 一次算一批组
  r["p"][k, m.idx["E01L"]], r["q_branch"][k], r["v"], r["Q_hub"]
python hydraulics.py   # 与 WNTR（EPANET emitter）逐组对比 30 个轮灌组
"""
import json
import pathlib
import numpy as np
import scipy.sparse as sp
import wntr
import network as net

HERE = pathlib.Path(__file__).parent
G = 9.81
Q_D, H_D, X_EMIT = 47.0, 10.0, 0.5
C_EMIT = Q_D / 3600 / H_D ** X_EMIT          # m3/s / m^x
P_B_CAP = 25.5 - 3.0                         # 支管入口稳态上限 = PE PN − 3 m 瞬态裕量
_curve = json.load(open(HERE / "valve_templates.json", encoding="utf-8"))["valve_curve"]
CURVE_S = np.array(_curve["opening_pct"]) / 100
CURVE_INVK = np.array(_curve["relative_inv_K"])


def cv_rel(s):
    """相对流量系数：先插值相对 1/K（与 TSNet 一致）再开方。"""
    return np.sqrt(np.interp(s, CURVE_S, CURVE_INVK))


def opening_from_cv(cv):
    """cv_rel 的反函数：相对流量系数 → 开度。"""
    return np.interp(np.asarray(cv) ** 2, CURVE_INVK, CURVE_S)


class SteadyModel:
    def __init__(self, inp=HERE / "benchmark71.inp", rough_mult=1.0):
        wn = wntr.network.WaterNetworkModel(str(inp))
        src = wn.reservoir_name_list[0]
        self.H0 = wn.get_node(src).base_head
        # 从水源 BFS 定向
        adj = {}
        for name, l in wn.links():
            adj.setdefault(l.start_node_name, []).append((name, l.end_node_name))
            adj.setdefault(l.end_node_name, []).append((name, l.start_node_name))
        self.nodes, self.links, parent_link, seen, stack = [], [], {}, {src}, [src]
        while stack:
            n = stack.pop()
            for l, m in adj.get(n, []):
                if m not in seen:
                    seen.add(m); parent_link[m] = l; stack.append(m)
                    self.nodes.append(m); self.links.append(l)
        self.idx = {n: i for i, n in enumerate(self.nodes)}
        self.lidx = {l: i for i, l in enumerate(self.links)}
        nN = len(self.nodes)
        # 链路 j 的下游节点就是 nodes[j]；路径矩阵 P[n, j] = 1 表示链路 j 在水源到 n 的路径上
        up = {self.links[i]: (wn.get_link(self.links[i]).start_node_name
                              if wn.get_link(self.links[i]).end_node_name == self.nodes[i]
                              else wn.get_link(self.links[i]).end_node_name) for i in range(nN)}
        rows, cols = [], []
        for i, n in enumerate(self.nodes):
            m = n
            while m != src:
                rows.append(i); cols.append(self.lidx[parent_link[m]]); m = up[parent_link[m]]
        self.P = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(nN, nN))
        self.elev = np.array([wn.get_node(n).elevation for n in self.nodes])
        L, D, eps, K, is_valve = [], [], [], [], []
        for l in self.links:
            o = wn.get_link(l)
            v = o.link_type == "Valve"
            is_valve.append(v); D.append(o.diameter)
            L.append(0.0 if v else o.length); eps.append(0.0 if v else o.roughness * rough_mult)   # wntr 已换算成 m
            K.append(float(o.initial_setting) if v else o.minor_loss)
        self.L, self.D, self.eps, self.K = map(np.array, (L, D, eps, K))
        self.is_valve = np.array(is_valve)
        self.A = np.pi * self.D ** 2 / 4
        br = [(v, e) for v, _, e in net.branches() if v in self.lidx]     # 其他 INP（如解析算例）里可能没有
        self.valves = [v for v, _ in br]
        self.v_pos = {v: i for i, v in enumerate(self.valves)}
        self.v_idx = np.array([self.lidx[v] for v, _ in br], int)
        self.e_idx = np.array([self.idx[e] for _, e in br], int)
        self.main = np.array([self.lidx[l] for l in (f"M{i}" for i in range(10)) if l in self.lidx], int)
        self.sub = np.array([self.lidx[l] for l in [f"P{x}{k}" for x in range(10) for k in range(1, 7)] +
                             [f"PA{x}" for x in range(10)] if l in self.lidx], int)
        self.src_links = np.array([self.lidx[l] for l, _ in adj[src]], int)

    def headloss(self, Q):
        """Q: (B, nL) m3/s → 各链路水头损失 (B, nL)，带符号。"""
        V = Q / self.A
        aV = np.abs(V)
        Re = np.maximum(aV * self.D / 1.004e-6, 1e-9)
        f_t = 0.25 / np.log10(self.eps / (3.7 * self.D) + 5.74 / Re ** 0.9) ** 2
        f = np.where(Re < 2000, 64 / Re, f_t)
        hf = np.where(self.is_valve, 0.0, f * self.L / self.D) * V * aV / (2 * G)
        return hf + self.Kc * V * aV / (2 * G)

    def _flow(self, C, q, iters, tol):
        for it in range(iters):
            Qlink = (self.P.T @ q.T).T
            H = self.H0 - (self.P @ self.headloss(Qlink).T).T
            q_new = C * np.maximum(H - self.elev, 0) ** X_EMIT
            err = np.abs(q_new - q).max()
            q = 0.5 * q + 0.5 * q_new
            if err < tol / 3600:
                break
        Qlink = (self.P.T @ q.T).T
        H = self.H0 - (self.P @ self.headloss(Qlink).T).T
        return q, Qlink, H, it + 1

    def solve(self, open_sets, extra_emitters=None, openings=None, closed_links=(), c_override=None, closed_per_set=None,
              p_cap=P_B_CAP, iters=200, tol=1e-6):
        """open_sets: B 个工况各自开启的支管阀集合。
        extra_emitters: B 个 {节点: C (m3/s/m^0.5)} 或 None（泄漏、爆管）。
        openings: {阀门: 开度}，部分开度卡死（不参与自动节流）。closed_links: 所有工况都关闭的链路（如隔离用的 VSx）。
        c_override: B 个 {节点: C}，替换默认滴头系数（与 TSNet 对比时用）。
        closed_per_set: B 个链路集合，只在对应工况关闭（卡开支管靠关分干管首端阀控制）。
        p_cap: 支管入口稳态压力上限，超过的支管阀自动节流；None 不节流。结果 opening (B, nV) 为各支管阀开度。"""
        B, nN, nV = len(open_sets), len(self.nodes), len(self.valves)
        closed = np.zeros((B, nN))
        for b, s in enumerate(open_sets):
            s = set(s)
            closed[b, self.v_idx] = [v not in s for v in self.valves]
        for l in closed_links:
            closed[:, self.lidx[l]] = 1
        for b, ls in enumerate(closed_per_set or []):
            for l in ls:
                closed[b, self.lidx[l]] = 1
        alive = (self.P @ closed.T).T == 0                     # 路径上无关闭阀门的节点
        self.Kc = np.tile(self.K, (B, 1))
        given = np.ones(nV)                                    # 卡死阀的给定开度
        for v, s in (openings or {}).items():
            self.Kc[:, self.lidx[v]] = self.K[self.lidx[v]] / max(cv_rel(s), 1e-6) ** 2
            if v in self.v_pos:
                given[self.v_pos[v]] = s
        fixed = given < 1
        C = np.zeros((B, nN))
        C[:, self.e_idx] = C_EMIT
        for b, co in enumerate(c_override or []):
            for n, c in (co or {}).items():
                C[b, self.idx[n]] = c
        for b, em in enumerate(extra_emitters or []):
            for n, c in (em or {}).items():
                C[b, self.idx[n]] += c
        C *= alive
        q = C * H_D ** X_EMIT                                  # 初值：设计压力下的出流
        K_base = self.Kc[:, self.v_idx].copy()
        K_thr = np.zeros((B, nV))
        Av2 = self.A[self.v_idx] ** 2
        for outer in range(40):
            q, Qlink, H, it = self._flow(C, q, iters, tol)
            if p_cap is None:
                break
            Qv = Qlink[:, self.v_idx]
            pB = H[:, self.v_idx] - self.elev[self.v_idx]      # 链路 j 的下游节点就是 nodes[j]，即阀后 B 节点
            on = (Qv > 1e-7) & ~fixed
            need = (pB - p_cap) * on
            active = on & ((need > 0.005) | (K_thr > 0))
            if np.abs(need * active).max() < 0.005:
                break
            dK = np.where(active, need * 2 * G * Av2 / np.maximum(Qv, 1e-9) ** 2, 0.0)
            K_thr = np.clip(K_thr + 0.8 * dK, 0.0, 1e5)
            self.Kc[:, self.v_idx] = K_base + K_thr
        cv = cv_rel(given) * np.sqrt(K_base / np.maximum(self.Kc[:, self.v_idx], 1e-12))
        opening = opening_from_cv(cv) * (closed[:, self.v_idx] == 0)
        p = np.where(alive, H - self.elev, np.nan)             # 隔离或关阀下游节点无压力意义
        return dict(p=p, H=H, Q=Qlink * 3600, v=np.abs(Qlink) / self.A, q_node=q * 3600,
                    q_branch=q[:, self.e_idx] * 3600, Q_hub=Qlink[:, self.src_links].sum(axis=1) * 3600,
                    alive=alive, opening=opening, iters=it)

def _wntr_check():
    import time
    wn = wntr.network.WaterNetworkModel(str(HERE / "benchmark71.inp"))
    groups = net.rotation_groups()
    m = SteadyModel()
    t0 = time.time(); r = m.solve(groups, p_cap=None); t_fast = time.time() - t0
    dp, dq, t_wntr = [], [], 0.0
    for g, valves in enumerate(groups):
        net.apply_group(wn, valves, q=0.0)
        for v, _, e in net.branches():
            wn.get_node(e).emitter_coefficient = C_EMIT if v in valves else 0.0
        t0 = time.time(); res = wntr.sim.EpanetSimulator(wn).run_sim(); t_wntr += time.time() - t0
        P = res.node["pressure"].iloc[0]
        ids = [n for n in m.nodes if r["alive"][g, m.idx[n]]]
        dp.append(np.abs(P[ids].values - r["p"][g, [m.idx[n] for n in ids]]).max())
        dq.append(abs(res.link["flowrate"].iloc[0]["M0"] * 3600 - r["Q_hub"][g]))
    print(f"30 组：快速求解 {t_fast*1000:.1f} ms（迭代 {r['iters']}），WNTR {t_wntr:.1f} s")
    print(f"最大压力偏差 {max(dp):.4f} m，首部流量最大偏差 {max(dq):.4f} m3/h")
    print("各组首部流量 %.0f–%.0f m3/h；支管实际流量 %.1f–%.1f m3/h" %
          (r["Q_hub"].min(), r["Q_hub"].max(), np.nanmin(np.where(r["q_branch"] > 0, r["q_branch"], np.nan)),
           r["q_branch"].max()))
    pe = r["p"][:, m.e_idx]
    print("支管末端压力（开启支管）%.1f–%.1f m" % (np.nanmin(np.where(r["q_branch"] > 0, pe, np.nan)), np.nanmax(np.where(r["q_branch"] > 0, pe, np.nan))))
    t0 = time.time(); r = m.solve(groups); t_thr = time.time() - t0
    pB = r["p"][:, m.v_idx]; on = r["opening"] > 0
    print("自动节流到 %.1f m：%d ms（外层迭代后），支管入口最大 %.2f m，节流阀 %d 个，开度 %.2f–%.2f，首部流量 %.0f–%.0f m3/h" % (
        P_B_CAP, t_thr * 1000, np.nanmax(pB[on]), int((r["opening"][on] < 0.999).sum()),
        r["opening"][on].min(), r["opening"][on & (r["opening"] < 0.999)].max() if (on & (r["opening"] < 0.999)).any() else 1,
        r["Q_hub"].min(), r["Q_hub"].max()))


if __name__ == "__main__":
    _wntr_check()
