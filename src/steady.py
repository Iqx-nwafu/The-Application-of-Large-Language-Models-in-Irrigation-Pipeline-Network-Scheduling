"""WNTR 稳态校核：30 个轮灌组逐组计算，干管损失按组流量与规范公式对比。

规范公式（GB/T 50485，硬 PVC）：hf = f·Q^m·L / d^b，Q 单位 L/h，d 单位 mm，
f = 0.464，m = 1.77，b = 4.77；工程上再加 10% 局部损失。
"""
import pathlib
import numpy as np
import pandas as pd
import wntr
import network as net

HERE = pathlib.Path(__file__).parent
PN_PVC, PN_PE = 61.2, 25.5         # 0.6 / 0.25 MPa 折算水头, m
H_DRIP = 10.0                      # 滴头工作水头, m


def spec_headloss(q_m3h, d_mm, L):
    return 0.464 * (q_m3h * 1000) ** 1.77 / d_mm ** 4.77 * L


def run_group(wn, valves):
    net.apply_group(wn, valves)
    return wntr.sim.EpanetSimulator(wn).run_sim()


def main():
    wn = wntr.network.WaterNetworkModel(str(HERE / "benchmark71.inp"))
    groups = net.rotation_groups()
    br = {v: (c, e) for v, c, e in net.branches()}
    rows, mains = [], []
    for g, valves in enumerate(groups, 1):
        r = run_group(wn, valves)
        P = r.node["pressure"].iloc[0]
        H = r.node["head"].iloc[0]
        Q = r.link["flowrate"].iloc[0] * 3600
        V = r.link["velocity"].iloc[0]
        x = int(valves[0][1])
        inlet = [f"C{v[1:]}" for v in valves]          # 支管阀前节点
        ends = [f"E{v[1:]}" for v in valves]
        rows.append(dict(
            group=g, submain=f"J{x}", valves="+".join(valves),
            Q_hub=Q["M0"], Q_submain=Q[f"PA{x}"],
            p_J0=P["J0"], p_Jx=P[f"J{x}"], p_J9=P["J9"],
            p_branch_inlet_min=P[inlet].min(), p_branch_inlet_max=P[inlet].max(),
            p_branch_end_min=P[ends].min(),
            v_main_max=V[[f"M{i}" for i in range(10)]].max(),
            v_submain_max=V[[f"P{x}{k}" for k in range(1, 7)]].max(),
            v_branch=V[[f"PB{v[1:]}" for v in valves]].mean(),
            p_min_all=P[[n for n in wn.junction_name_list]].min(),
            p_max_all=P[[n for n in wn.junction_name_list]].max(),
        ))
        # 干管各段损失 / 100 m 与规范对比（只记录有流量的段）
        for i in range(10):
            L = net.MAIN_LEN[i]
            q = Q[f"M{i}"]
            if q < 1:
                continue
            hf = H["Hub" if i == 0 else f"J{i-1}"] - H[f"J{i}"]
            mains.append(dict(group=g, pipe=f"M{i}", Q_m3h=q, L=L,
                              hf_wntr_per100=hf / L * 100,
                              hf_spec_per100=spec_headloss(q, net.D_MAIN, 100),
                              hf_spec_x1_1_per100=1.1 * spec_headloss(q, net.D_MAIN, 100)))
    df = pd.DataFrame(rows)
    dm = pd.DataFrame(mains)
    df.to_csv(HERE / "steady_groups.csv", index=False, float_format="%.3f")
    dm.to_csv(HERE / "steady_main_headloss.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(df.round(2).to_string(index=False))
    print("\n干管损失/100 m（组流量 ≈ %.0f m3/h）: WNTR %.3f  规范 %.3f  规范×1.1 %.3f  论文表4 0.0438（按 47 m3/h）"
          % (dm.Q_m3h.max(), dm.hf_wntr_per100.max(), dm.hf_spec_per100.max(), dm.hf_spec_x1_1_per100.max()))
    print("支管入口压力范围 %.1f–%.1f m（PE 支管 PN 0.25 MPa ≈ %.1f m）" % (df.p_branch_inlet_min.min(), df.p_branch_inlet_max.max(), PN_PE))
    print("支管末端最低压力 %.1f m（滴头工作水头 %.0f m）" % (df.p_branch_end_min.min(), H_DRIP))
    print("干管/分干管最大流速 %.2f / %.2f m/s" % (df.v_main_max.max(), df.v_submain_max.max()))
    print("全网压力极值 %.1f–%.1f m（PVC PN 0.6 MPa ≈ %.1f m）" % (df.p_min_all.min(), df.p_max_all.max(), PN_PVC))


if __name__ == "__main__":
    main()
