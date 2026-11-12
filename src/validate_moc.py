"""E0 平台验证：moc.py 对解析解（Joukowsky）和对 TSNet（同一管网关阀）。

1. 解析解：水库–单管 1000 m（a = 1000 m/s）–阀门，阀门 0.02 s 内关闭（< 2L/a），阀前升压应为 a·V0/g。
2. TSNet 对比：组 1（J01+J02）运行，四个支管阀 t = 2 s 起 2 s 线性关闭。
   两边设定一致：支管末端按需水驱动稳态（47 m3/h）标定孔口系数；dt = 0.01 s；通用闸阀曲线；
   TSNet 的阀门全开 K = 0.2（按管道流速），所以对比用的 INP 把阀门 setting 换成等效值。
   爆管不与 TSNet 对比：TSNet 0.3.1 的 add_burst 在爆管开始前就出现升压尖峰，结果不可信。
指标：最大压力相对误差、波到达时间误差、压力时程 RMSE。

python validate_moc.py   # TSNet 约 3 min
"""
import pathlib
import time
import numpy as np
import pandas as pd
import wntr
import network as net

HERE = pathlib.Path(__file__).parent
G1 = net.rotation_groups()[0]
TF, T0, TC = 30.0, 2.0, 2.0
NODES = ["J0", "J5", "J01", "J02", "J03", "J04", "J05", "J06", "C01L", "C02R"]


VAL_INP = HERE / "val_bench.inp"


def write_val_inp():
    """阀门 setting 换成与 TSNet 一致的全开损失：K_pipe = 0.2 → K_valve = 0.2·(A_v/A_pipe)²。"""
    k_br = 0.2 * (net.D_VBR / net.D_BRANCH) ** 4
    k_sub = 0.2 * (net.D_VSUB / net.D_SUB) ** 4
    txt = open(HERE / "benchmark71.inp").read()
    txt = txt.replace(" TCV 0.5 ", f" TCV {k_br:.4f} ").replace(" TCV 0.3 ", f" TCV {k_sub:.4f} ")
    open(VAL_INP, "w").write(txt)


def dd_coeffs():
    wn = wntr.network.WaterNetworkModel(str(VAL_INP))
    net.apply_group(wn, G1)
    p = wntr.sim.EpanetSimulator(wn).run_sim().node["pressure"].iloc[0]
    return {f"E{v[1:]}": net.Q_BRANCH / 3600 / np.sqrt(p[f"E{v[1:]}"]) for v in G1}


def run_tsnet(case):
    import tsnet
    import transient  # noqa: F401  兼容补丁
    tm = tsnet.network.TransientModel(str(VAL_INP))
    tm.set_wavespeed([transient.A_PE if p.startswith(("PC", "PB")) else transient.A_PVC for p in tm.pipe_name_list])
    tm.set_time(TF)
    for x in range(10):
        tm.get_link(f"VS{x}").initial_status = wntr.network.LinkStatus.Open
    for v in G1:
        tm.valve_closure(v, [TC, T0, 0, 1])
    tm = tsnet.simulation.Initializer(tm, 0, "DD")
    t0 = time.time()
    tm = tsnet.simulation.MOCSimulator(tm, str(HERE / f"val_tsnet_{case}"), "steady")
    cpu = time.time() - t0
    t = np.arange(len(tm.get_node("J0").head)) * tm.time_step
    return case, cpu, tm.time_step, pd.DataFrame({"t": t, **{n: tm.get_node(n).head - tm.get_node(n).elevation for n in NODES}})


def run_moc(case, dt):
    from moc import TransientModel
    sim = TransientModel(VAL_INP, dt=dt)
    co = dd_coeffs()
    t0 = time.time()
    acts = [dict(valve=v, op="close", t_start_s=T0, template="close_linear", params=dict(duration_s=TC)) for v in G1]
    out = sim.run(G1, acts, tf=TF, c_override=co, series_nodes=NODES, p_cap=None)
    cpu = time.time() - t0
    return cpu, pd.DataFrame({"t": out["t"], **out["series"]})


def arrival(t, p, tol=0.2):
    i = np.argmax(np.abs(p - p[0]) > tol)
    return t[i] if i else np.nan


def joukowsky():
    """水库 H0 = 50 m – 管道 1000 m, D 0.2 m, a = 1000 m/s – 阀 – 6 m 短管 – 出流节点（大孔口）。"""
    import moc
    inp = HERE / "val_single.inp"
    open(inp, "w").write("""[JUNCTIONS]
N1 0 0
N2 0 0
N3 0 0
[RESERVOIRS]
R1 50
[PIPES]
P1 R1 N1 1000 200 0.01 0 Open
P2 N2 N3 6 200 0.01 0 Open
[VALVES]
V1 N1 N2 200 TCV 0.2 0
[EMITTERS]
N3 100
[OPTIONS]
Units CMH
Headloss D-W
[END]
""")
    a_old = moc.A_PVC
    moc.A_PVC = 1000.0
    sim = moc.TransientModel(inp, dt=0.002)
    moc.A_PVC = a_old
    co = {"N3": 100 / 3600}
    act = [dict(valve="V1", op="close", t_start_s=0.1, template="close_linear", params=dict(duration_s=0.02))]
    out = sim.run([], act, tf=3.0, c_override=co, series_nodes=["N1"], p_cap=None)
    V0 = out["steady"]["Q"][0, sim.steady.lidx["P1"]] / 3600 / (np.pi * 0.2 ** 2 / 4)
    p0 = out["series"]["N1"][0]
    k = np.searchsorted(out["t"], 0.15)                    # 关闭完成后、反射波回来前（2L/a = 2 s）
    rise = out["series"]["N1"][k] - p0
    jk = sim.a[sim.pipes.index("P1")] * V0 / 9.81
    sN = out["series"]["N1"]
    t_ref = out["t"][k + np.argmax(sN[k:] < p0 + rise / 2)]     # 水库反射的降压波到达
    print(f"Joukowsky：V0 = {V0:.3f} m/s，解析首波升压 {jk:.2f} m，MOC {rise:.2f} m，相对误差 {(rise - jk) / jk:+.2%}；"
          f"之后的继续升压 {out['series']['N1'].max() - p0 - rise:.1f} m 为管道摩阻引起的 line packing；"
          f"降压波到达 {t_ref:.2f} s（理论约 0.11 + 2L/a = 2.11 s）")
    return dict(V0=V0, analytic=jk, moc=rise)


def main():
    jk = joukowsky()
    write_val_inp()
    case, cpu, dt, df = run_tsnet("close")
    ts = {case: (cpu, dt, df)}
    rows = []
    for case, (cpu_ts, dt, a) in ts.items():
        cpu_m, b = run_moc(case, dt)
        b = b.set_index("t").reindex(a.t, method="nearest").reset_index()
        for n in NODES:
            pa, pb = a[n].values, b[n].values
            m = np.isfinite(pa)
            rows.append(dict(case=case, node=n, pmax_tsnet=pa[m].max(), pmax_moc=pb[m].max(),
                             pmax_rel_err=(pb[m].max() - pa[m].max()) / pa[m].max(),
                             pmin_tsnet=pa[m].min(), pmin_moc=pb[m].min(),
                             arrival_err_s=arrival(b.t.values, pb) - arrival(a.t.values, pa),
                             rmse=np.sqrt(np.mean((pa[m] - pb[m]) ** 2)), cpu_tsnet=cpu_ts, cpu_moc=cpu_m))
        pd.DataFrame({"t": a.t, **{f"tsnet_{n}": a[n] for n in NODES}, **{f"moc_{n}": b[n] for n in NODES}}) \
            .to_csv(HERE / f"val_{case}_series.csv", index=False, float_format="%.3f")
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "val_moc_vs_tsnet.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    fig, ax = plt.subplots(1, 1, figsize=(9, 3.5), squeeze=False); ax = ax[:, 0]
    for i, case in enumerate(ts):
        d = pd.read_csv(HERE / f"val_{case}_series.csv")
        for n, c in [("J02", "C0"), ("J05", "C1"), ("J0", "C2")]:
            ax[i].plot(d.t, d[f"tsnet_{n}"], c=c, lw=1.6, alpha=0.5, label=f"TSNet {n}")
            ax[i].plot(d.t, d[f"moc_{n}"], c=c, lw=0.8, ls="--", label=f"moc {n}")
        ax[i].set_title("组 1 四阀 2 s 线性关闭"); ax[i].set_ylabel("压力水头 m")
        ax[i].legend(ncol=3, fontsize=8)
    ax[-1].set_xlabel("t / s")
    fig.tight_layout(); fig.savefig(HERE / "val_moc_vs_tsnet.png", dpi=130)


if __name__ == "__main__":
    main()
