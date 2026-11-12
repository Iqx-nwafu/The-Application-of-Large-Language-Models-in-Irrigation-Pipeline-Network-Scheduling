"""71 节点基准管网（呼图壁联丰村二队 2 号系统，论文原方案拓扑）。

节点命名
  Hub            定水头水库，HGL 25 m（71 节点中的首部）
  J0..J9         干管节点，高程 0
  Jx1..Jx6       分干管节点（x = 0..9），高程沿分干管线性降到 -3.5 m
  Sx             分干管首端阀 VSx 的阀后节点
  BxkL / BxkR    支管阀 Vxk{L,R} 的阀后节点
  ExkL / ExkR    支管末端（出流节点，设计流量 47 m3/h）
每个阀门前后各有一段短管（TSNet 要求阀门两端都是管道）。
"""
import wntr

H_SOURCE = 25.0                                   # 水源水面 HGL, m
MAIN_LEN = [50.0] + [100.0] * 9                   # Hub-J0, J0-J1, ..., J8-J9（论文）
SUB_LEN = [34.0, 94.0, 94.0, 120.0, 94.0, 94.0]   # Jx-Jx1, Jx1-Jx2, ...（实施方案）
SUB_DROP = 3.5                                    # 分干管全长高程降, m
BRANCH_LEN = 60.0
STUB = 6.0                                        # 阀前短管长度, m

D_MAIN, D_SUB, D_BRANCH = 235.4, 150.4, 86.0      # 内径, mm
D_VSUB, D_VBR = 150.0, 80.0                       # 阀门口径, mm
K_OPEN = {"sub": 0.3, "branch": 0.5}              # 全开局部损失系数（蝶阀通用值）
ROUGH = 0.01                                      # Darcy-Weisbach 绝对粗糙度, mm（PVC-U / PE）
Q_BRANCH = 47.0                                   # 支管设计流量, m3/h

SIDES = "LR"


def sub_node_elev():
    """分干管节点高程：按距 Jx 的累计长度线性下降。"""
    total, s, out = sum(SUB_LEN), 0.0, []
    for L in SUB_LEN:
        s += L
        out.append(-SUB_DROP * s / total)
    return out


def branches():
    """全部 120 条支管：(阀门名, 分干管节点, 支管末端节点)。"""
    return [(f"V{x}{k}{s}", f"J{x}{k}", f"E{x}{k}{s}")
            for x in range(10) for k in range(1, 7) for s in SIDES]


def rotation_groups():
    """实际轮灌表：30 组，每组同一分干管相邻两节点的 4 条支管，每组 4 h。"""
    groups = []
    for x in range(10):
        for a, b in [(1, 2), (3, 4), (5, 6)]:
            groups.append([f"V{x}{k}{s}" for k in (a, b) for s in SIDES])
    return groups


def build_inp(path):
    """写出 INP（默认工况：第 1 组 J01+J02 运行，其余支管阀关闭）。"""
    elev = sub_node_elev()
    J, P, V = [], [], []          # (name, elev, demand) / (name, n1, n2, L, D) / (name, n1, n2, D, K)
    J += [(f"J{i}", 0.0) for i in range(10)]
    P.append(("M0", "Hub", "J0", MAIN_LEN[0], D_MAIN))
    P += [(f"M{i}", f"J{i-1}", f"J{i}", MAIN_LEN[i], D_MAIN) for i in range(1, 10)]
    for x in range(10):
        # 分干管首端阀：Jx -短管- Ax -阀- Sx -管- Jx1
        J += [(f"A{x}", 0.0), (f"S{x}", 0.0)]
        P.append((f"PA{x}", f"J{x}", f"A{x}", STUB, D_SUB))
        V.append((f"VS{x}", f"A{x}", f"S{x}", D_VSUB, K_OPEN["sub"]))
        up = f"S{x}"
        for k in range(1, 7):
            n = f"J{x}{k}"
            J.append((n, elev[k - 1]))
            L = SUB_LEN[0] - STUB if k == 1 else SUB_LEN[k - 1]
            P.append((f"P{x}{k}", up, n, L, D_SUB))
            up = n
            for s in SIDES:
                # 支管：Jxk -短管- Cxks -阀- Bxks -管- Exks
                J += [(f"C{x}{k}{s}", elev[k - 1]), (f"B{x}{k}{s}", elev[k - 1]), (f"E{x}{k}{s}", elev[k - 1])]
                P.append((f"PC{x}{k}{s}", n, f"C{x}{k}{s}", STUB, D_BRANCH))
                V.append((f"V{x}{k}{s}", f"C{x}{k}{s}", f"B{x}{k}{s}", D_VBR, K_OPEN["branch"]))
                P.append((f"PB{x}{k}{s}", f"B{x}{k}{s}", f"E{x}{k}{s}", BRANCH_LEN - STUB, D_BRANCH))

    active = set(rotation_groups()[0])
    demand = {e: Q_BRANCH for v, _, e in branches() if v in active}
    L = ["[TITLE]", "Hutubi Lianfeng No.2 system, 71-node benchmark (paper original layout)",
         "Default state: rotation group 1 (J01+J02) open; use network.apply_group() for others", "",
         "[JUNCTIONS]", ";ID  Elev  Demand"]
    L += [f"{n:<8} {z:8.3f} {demand.get(n, 0.0):8.2f}" for n, z in J]
    L += ["", "[RESERVOIRS]", f"Hub {H_SOURCE:.2f}", "", "[PIPES]", ";ID n1 n2 Length Diam Rough Minor Status"]
    L += [f"{p:<8} {a:<6} {b:<6} {l:7.1f} {d:6.1f} {ROUGH} 0 Open" for p, a, b, l, d in P]
    L += ["", "[VALVES]", ";ID n1 n2 Diam Type Setting Minor"]
    L += [f"{v:<8} {a:<6} {b:<6} {d:6.1f} TCV {k} 0" for v, a, b, d, k in V]
    L += ["", "[STATUS]"]
    L += [f"{v} Closed" for v, *_ in V if v.startswith("V") and not v.startswith("VS") and v not in active]
    L += ["", "[OPTIONS]", "Units CMH", "Headloss D-W", "Accuracy 0.0001", "Trials 200", "",
          "[TIMES]", "Duration 0", "", "[END]"]
    open(path, "w").write("\n".join(L) + "\n")


def apply_group(wn, valves_open, q=Q_BRANCH):
    """把 wn 设成「只开 valves_open 这些支管阀」的稳态工况。"""
    valves_open = set(valves_open)
    for v, _, e in branches():
        on = v in valves_open
        # TCV 设 Active 才按 setting 计局部损失；设 Open 时 EPANET 忽略 setting
        wn.get_link(v).initial_status = wntr.network.LinkStatus.Active if on else wntr.network.LinkStatus.Closed
        wn.get_node(e).demand_timeseries_list[0].base_value = q / 3600.0 if on else 0.0
    return wn


if __name__ == "__main__":
    import pathlib
    out = pathlib.Path(__file__).with_name("benchmark71.inp")
    build_inp(out)
    wn = wntr.network.WaterNetworkModel(str(out))
    print(out.name, "junctions", wn.num_junctions, "pipes", wn.num_pipes, "valves", wn.num_valves)
