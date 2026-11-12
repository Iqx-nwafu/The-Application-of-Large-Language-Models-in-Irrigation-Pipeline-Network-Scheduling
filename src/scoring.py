"""方案 A 的目标标量化（作者 2026-10-27 定）：缺水量字典序优先，其余 CV_q、T_total、V_over 用熵权 TOPSIS。

参考集：标称场景 NSGA-II 前沿合并后的字典序可行解（optimizer.py reference），熵权 w 和各目标 min–max 范围
一次算定存 topsis_reference.json；之后所有方案（优化器、模型、基线）都用同一 w 和范围算贴近度 C*，可直接比较。
归一化：r_j = clip(1 − (x_j − min_j)/(max_j − min_j), 0, 1)，三项都越小越好；理想解 r = 1，负理想解 r = 0；
C* = d⁻ / (d⁺ + d⁻)，d 为加权欧氏距离。
字典序：缺水量 ≤ D_min + max(1 m³, 0.2% 总需水) 的方案才进入 TOPSIS 比较。
敏感性：每个权重分别 ×0.5、×1.5（其余按比例归一）和等权，看选解是否变化。
能耗 E_pump 在定水头下与超灌水量线性相关，不单列为目标，只在评价表里报告。
"""
import json
import pathlib
import numpy as np

HERE = pathlib.Path(__file__).parent
REF = HERE / "topsis_reference.json"
OBJ = ["cv_q", "T_total_h", "over_m3"]
_ref = None


def reference():
    global _ref
    if _ref is None and REF.exists():
        _ref = json.load(open(REF, encoding="utf-8"))
    return _ref


def entropy_weights(F):
    """F (n, m)，越小越好。熵权：信息熵小（离散度大）的目标权重大。"""
    F = np.asarray(F, float)
    X = (F.max(0) - F) / np.maximum(F.max(0) - F.min(0), 1e-12)
    P = (X + 1e-9) / (X + 1e-9).sum(0)
    E = -(P * np.log(P)).sum(0) / np.log(max(len(F), 2))
    d = 1 - E
    return d / d.sum()


def closeness_matrix(F, w, lo, hi):
    F, w, lo, hi = (np.asarray(x, float) for x in (F, w, lo, hi))
    r = np.clip(1 - (F - lo) / np.maximum(hi - lo, 1e-12), 0, 1)
    v = r * w
    dp = np.linalg.norm(v - w, axis=1)
    dm = np.linalg.norm(v, axis=1)
    return dm / np.maximum(dp + dm, 1e-12)


def closeness(obj, ref=None):
    """单个方案的 C*；没有参考集时返回 None。"""
    ref = ref or reference()
    if ref is None:
        return None
    return round(float(closeness_matrix([[obj[k] for k in OBJ]], ref["weights"], ref["min"], ref["max"])[0]), 4)


def d_tol(D_total):
    return max(1.0, 0.002 * D_total)


def sensitivity(F, w, lo, hi):
    w = np.asarray(w, float)
    cases = {"equal": np.ones(len(w)) / len(w)}
    for j in range(len(w)):
        for f in (0.5, 1.5):
            ww = w.copy(); ww[j] *= f
            cases[f"{OBJ[j]}x{f}"] = ww / ww.sum()
    return {k: int(np.argmax(closeness_matrix(F, ww, lo, hi))) for k, ww in cases.items()}


def select(results, ref=None):
    """results: evaluate() 结果列表。可行 → 缺水字典序 → C* 最大。返回 (索引, 选解信息)。"""
    feas = [i for i, r in enumerate(results) if r["feasible"]]
    pool = feas or list(range(len(results)))
    d = np.array([results[i]["objectives"]["deficit_m3"] for i in pool])
    tol = d_tol(results[pool[0]]["D_total"])
    lexi = [i for i, di in zip(pool, d) if di <= d.min() + tol]
    ref = ref or reference()
    F = np.array([[results[i]["objectives"][k] for k in OBJ] for i in lexi])
    if ref is None:                                      # 无参考集：用本前沿的熵权和范围
        w, lo, hi = entropy_weights(F), F.min(0), F.max(0)
    else:
        w, lo, hi = ref["weights"], ref["min"], ref["max"]
    C = closeness_matrix(F, w, lo, hi)
    k = int(np.argmax(C))
    sens = sensitivity(F, w, lo, hi)
    return lexi[k], dict(base=lexi[k], n_feasible=len(feas), n_lexi=len(lexi), deficit_min=round(float(d.min()), 1), deficit_tol=round(tol, 1),
                         weights=[round(float(x), 4) for x in np.asarray(w)], reference=ref is not None,
                         C_best=round(float(C[k]), 4),
                         sensitivity=dict(stable=all(s == k for s in sens.values()), chosen={n: lexi[s] for n, s in sens.items()}))


def build_reference(fronts, note=""):
    """fronts: 各标称场景字典序筛选后的目标矩阵 → 合并算熵权与范围并保存。"""
    F = np.vstack(fronts)
    ref = dict(objectives=OBJ, weights=[round(float(x), 4) for x in entropy_weights(F)],
               min=[round(float(x), 4) for x in F.min(0)], max=[round(float(x), 4) for x in F.max(0)], n=int(len(F)), note=note)
    json.dump(ref, open(REF, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    global _ref
    _ref = ref
    return ref
