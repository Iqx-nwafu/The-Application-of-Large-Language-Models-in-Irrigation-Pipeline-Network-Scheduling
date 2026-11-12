"""TSNet 瞬态初算：轮灌组 1（J01+J02）切换到组 2（J03+J04）。

三个切换方案
  fast     关 4 开 4 同时动作，历时 2 s（电磁阀直接动作）
  linear   先开后关，各 10 s 线性，错开 6 s
  twostage 先开后关；关阀两阶段：前 3 s 关到 30% 开度，再用 12 s 关完
阀门开度–流量系数用 TSNet 默认闸阀曲线。
"""
import pathlib
import time
import numpy as np
import pandas as pd
import wntr
import tsnet
import network as net

HERE = pathlib.Path(__file__).parent


def _cal_N(tm, dt):
    """tsnet 0.3.1 的 cal_N 返回 (n,1) 数组，numpy 2 下 int() 会报错；这里返回一维。"""
    N = np.zeros(tm.num_pipes)
    for _, pipe in tm.pipes():
        N[int(pipe.id) - 1] = round(int(pipe.length / (pipe.wavev * dt)))
    return N


def _adjust_wavev(tm):
    """同上：原版 theta 是 (1,1) 数组，time_step 变成数组。最小二乘波速调整，逻辑不变。"""
    phi = np.array([pipe.length / pipe.wavev / pipe.number_of_segments for _, pipe in tm.pipes()], dtype=np.longdouble)
    tm.wavespeed_adj = float(np.sum(phi ** 2))
    theta = float(phi.sum() / (phi ** 2).sum())
    for _, pipe in tm.pipes():
        pipe.wavev = float(pipe.wavev * phi[int(pipe.id) - 1] * theta)
    tm.time_step = 1.0 / theta
    return tm


def _cal_roughness_coef(pipe, V, hl):
    """原版把无流量管段的 D-W 摩阻系数设为 0，wntr 1.5 不允许；无流量管段取 f = 0.02。"""
    if abs(V) >= 1e-5 and hl >= 1e-3:
        pipe.roughness = hl / (pipe.length / pipe.diameter) / (V ** 2 / 2 / 9.8)
    else:
        pipe.roughness = 0.02
    if pipe.roughness > 0.08:
        pipe.roughness = 0.03
    Re = getattr(pipe, "initial_Re", 0)
    pipe.roughness_height = max(10 ** (-1 / 1.8 / np.sqrt(pipe.roughness)) - 6.9 / Re, 0) if Re > 0 else 0
    return pipe


tsnet.simulation.initialize.cal_roughness_coef = _cal_roughness_coef
tsnet.network.discretize.cal_N = _cal_N
tsnet.network.discretize.adjust_wavev = _adjust_wavev
A_PVC, A_PE = 300.0, 100.0           # 波速, m/s（PVC-U 干管/分干管, 低密度 PE 支管）
TF = 60.0                            # 模拟时长, s
PN_PVC, PN_PE, H_VAC = 61.2, 25.5, -5.0

G1, G2 = net.rotation_groups()[0], net.rotation_groups()[1]


def two_stage(tm, t0, t1, s1, t2):
    """两阶段关阀开度序列：t0 起 t1 秒内线性关到开度 s1，再 t2 秒关到 0。"""
    t = np.arange(tm.time_step * 0 , tm.simulation_period + tm.time_step, tm.time_step)
    s = np.ones_like(t)
    a = (t >= t0) & (t < t0 + t1)
    s[a] = 1 - (1 - s1) * (t[a] - t0) / t1
    b = (t >= t0 + t1) & (t < t0 + t1 + t2)
    s[b] = s1 * (1 - (t[b] - t0 - t1) / t2)
    s[t >= t0 + t1 + t2] = 0.0
    return s


def build(scenario):
    inp = str(HERE / "benchmark71.inp")
    # 组 2 稳态压力，用于给组 2 支管末端一个与稳态一致的出流系数
    wn = wntr.network.WaterNetworkModel(inp)
    net.apply_group(wn, G2)
    p2 = wntr.sim.EpanetSimulator(wn).run_sim().node["pressure"].iloc[0]

    tm = tsnet.network.TransientModel(inp)
    a = [A_PE if p.startswith(("PC", "PB")) else A_PVC for p in tm.pipe_name_list]
    tm.set_wavespeed(a)
    tm.set_time(TF)
    dt = tm.time_step

    t_open, t_close = 2.0, 2.0
    if scenario == "fast":
        for v in G2: tm.valve_opening(v, [2.0, t_open, 1, 1])
        for v in G1: tm.valve_closure(v, [2.0, t_close, 0, 1])
    elif scenario == "linear":
        t_close = 8.0
        for v in G2: tm.valve_opening(v, [10.0, t_open, 1, 1])
        for v in G1: tm.valve_closure(v, [10.0, t_close, 0, 1])
    elif scenario == "twostage":
        t_close = 8.0
        for v in G2: tm.valve_opening(v, [10.0, t_open, 1, 1])
        for v in G1:
            tm.valve_closure(v, [15.0, t_close, 0, 1])
            tm.get_link(v).operation_rule = two_stage(tm, t_close, 3.0, 0.3, 12.0)

    # 分干管首端阀是 TCV，EPANET 状态为 Active；TSNet 只认 Open/Closed，不设则按关闭处理
    for x in range(10):
        tm.get_link(f"VS{x}").initial_status = wntr.network.LinkStatus.Open

    tm = tsnet.simulation.Initializer(tm, 0, "DD")
    # 组 2 支管：阀后初始为零压满水、无流量；滴头出流系数按组 2 稳态压力标定，
    # 开阀后随压力建立自然出流（TSNet 对负压节点按止回处理）
    for v in G2:
        e = f"E{v[1:]}"
        node = tm.get_node(e)
        node.demand_coeff = net.Q_BRANCH / 3600 / np.sqrt(max(p2[e], 1.0))
        node.base_demand_coeff = node.demand_coeff
        pipe = tm.get_link(f"PB{v[1:]}")
        pipe.initial_head = np.full_like(pipe.initial_head, node.elevation)
        pipe.initial_velocity = np.zeros_like(pipe.initial_velocity)
    return tm, dt


def run(scenario):
    tm, dt = build(scenario)
    t0 = time.time()
    tm = tsnet.simulation.MOCSimulator(tm, str(HERE / f"tr_{scenario}"), "steady")
    cpu = time.time() - t0
    t = np.arange(len(tm.get_node("J0").head)) * dt
    nodes = {"main": [f"J{i}" for i in range(10)],
             "submain_J0": [f"J0{k}" for k in range(1, 7)],
             "branch_G1": [f"B{v[1:]}" for v in G1] + [f"E{v[1:]}" for v in G1],
             "branch_G2": [f"B{v[1:]}" for v in G2] + [f"E{v[1:]}" for v in G2]}
    summary = {}
    for k, ns in nodes.items():
        H = np.array([tm.get_node(n).head - tm.get_node(n).elevation for n in ns])
        summary[k] = (H.max(), H.min(), ns[H.max(axis=1).argmax()], ns[H.min(axis=1).argmin()])
    series = pd.DataFrame({"t": t, **{n: tm.get_node(n).head - tm.get_node(n).elevation
                                    for n in ["J0", "J02", "J04", "C01L", "B01L", "B03L", "E01L", "E03L"]}})
    series.to_csv(HERE / f"tr_{scenario}.csv", index=False, float_format="%.3f")
    return summary, cpu, dt


def main():
    import sys
    scenarios = sys.argv[1:] or ["fast", "linear", "twostage"]
    out = []
    for sc in scenarios:
        s, cpu, dt = run(sc)
        print(f"\n[{sc}] dt={dt:.4f} s, 模拟 {TF:.0f} s, CPU {cpu:.1f} s")
        for k, (hmax, hmin, nmax, nmin) in s.items():
            print(f"  {k:<12} 最大 {hmax:6.1f} m @{nmax:<5}  最小 {hmin:6.1f} m @{nmin}")
            out.append(dict(scenario=sc, region=k, p_max=hmax, node_max=nmax, p_min=hmin, node_min=nmin, cpu_s=cpu))
    df = pd.DataFrame(out)
    df.to_csv(HERE / "transient_summary.csv", index=False, float_format="%.2f")
    print(f"\n阈值：PVC PN {PN_PVC} m，PE 支管 PN {PN_PE} m，H_vac {H_VAC} m")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    fig, ax = plt.subplots(len(scenarios), 1, figsize=(9, 3 * len(scenarios)), sharex=True, squeeze=False); ax = ax[:, 0]
    for i, sc in enumerate(scenarios):
        d = pd.read_csv(HERE / f"tr_{sc}.csv")
        for n in ["J0", "J02", "B01L", "B03L"]:
            ax[i].plot(d.t, d[n], label=n, lw=1)
        ax[i].axhline(PN_PVC, ls="--", c="k", lw=0.8); ax[i].axhline(PN_PE, ls=":", c="k", lw=0.8); ax[i].axhline(H_VAC, ls="--", c="r", lw=0.8)
        ax[i].set_title(sc); ax[i].set_ylabel("压力水头 m"); ax[i].legend(ncol=4, fontsize=8)
    ax[-1].set_xlabel("t / s")
    fig.tight_layout(); fig.savefig(HERE / "transient_switch.png", dpi=130)


if __name__ == "__main__":
    main()
