"""训练用切换应力场景（作者 2026-11-02 13:16 NZDT 同意训练集加少量切换应力场景，份额与费用待报批）。

主实验与 plan_5000 的切换都是分布内相邻组切换，照提示默认动作做就安全，学生见不到瞬态越限、学不到瞬态自检。
这里用 make_hard.py 切换族的同一套构造（跨分干管、高水头、高波速），但与 hard50 完全分开：
  源场景只取 split == train 的标称场景（需水、糙率）；
  切换组对不用 hard50 的 22 对，也不碰分干管 J7（分布外留出位置）；
  物理参数在 波速 ×[1.0, 1.5]、水源水头 [28, 35] m 内连续抽样（hard50 只用 5 个离散点，这里与之不重合）。
入选 A 档：提示默认动作也越限（默认 ≥ 1%，hard50 用 3%；必须靠瞬态校核与反馈改慢），且兜底逐档放慢有解；每个组对最多 --per-pair 个。
排除 hard50 的组对后 A 档只占候选约 9%（hard50 网格 27%），所以每个组对抽 --draws 组参数。

python distill/make_stress.py --n 200   → distill/scenarios_stress.jsonl、distill/plans/plan_stress.jsonl、distill/plans/stress_probe.csv
python distill/make_stress.py --n 200 --select-only   # 只按已有 stress_probe.csv 重选
"""
import argparse
import json
import os
import pathlib
import sys
for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(k, "1")
import numpy as np
from multiprocessing import Pool

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import make_hard as mh                                       # noqa: E402

SEED = 20261102


def candidates(draws):
    S = [json.loads(l) for l in open(ROOT / "scenarios.jsonl", encoding="utf-8")]
    nom = [s for s in S if s["kind"] == "nominal" and s["split"] == "train"]
    hard = {tuple(json.loads(l)["switch_pair"]) for l in open(ROOT / "scenarios_hard.jsonl", encoding="utf-8") if "switch_pair" in l}
    pairs = [(3 * x + t, 3 * y) for x in range(10) for y in range(10) for t in (1, 2)
             if x != y and mh.HOLDOUT not in (x, y) and (3 * x + t, 3 * y) not in hard]
    rng = np.random.default_rng(SEED)
    out = []
    for k1, k2 in pairs:
        for _ in range(draws):
            src = nom[int(rng.integers(len(nom)))]
            w, h = round(float(rng.uniform(1.0, 1.5)), 3), round(float(rng.uniform(28.0, 35.0)), 2)
            pert = dict(src["perturbation"], wavespeed_mult=w, h_source=h, stress_test=True)
            out.append(dict(id=f"T{len(out):05d}", kind="nominal", split="train_stress", irrigation_event=src["irrigation_event"],
                            quota_m3_per_mu=src["quota_m3_per_mu"], window=src["window"], perturbation=pert, demand_m3=src["demand_m3"],
                            switch_pair=[k1, k2], source=src["id"]))
    return out, len(pairs), len(hard)


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--per-pair", type=int, default=8)
    ap.add_argument("--draws", type=int, default=32, help="每个组对抽几组物理参数")
    ap.add_argument("--min-excess", type=float, default=1.01, help="提示默认动作的最坏压力比下限")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--select-only", action="store_true")
    a = ap.parse_args()
    cand, n_pairs, n_hard = candidates(a.draws)
    print(f"候选 {len(cand)}（{n_pairs} 个组对 × {a.draws}，已排除 hard50 的 {n_hard} 对与 J7）")
    probe_path = HERE / "plans" / "stress_probe.csv"
    if a.select_only:
        rs = pd.read_csv(probe_path).to_dict("records")
    else:
        with Pool(a.workers) as p:
            rs = p.map(mh.eval_switch, cand, chunksize=4)
        pd.DataFrame(rs).to_csv(probe_path, index=False, encoding="utf-8-sig")
    ok = [r for r in rs if r["baseline_viol"] and r["solvable"] and r["tier"] == "A" and r["prompt_default_worst_ratio"] >= a.min_excess]
    sel = mh.greedy(ok, a.n, lambda r: r["prompt_default_worst_ratio"], caps=[(lambda r: (int(r["g1"]), int(r["g2"])), a.per_pair)])
    print(f"A 档且最坏压力比 ≥ {a.min_excess}：{len(ok)}；入选 {len(sel)}（组对 {len({(int(r['g1']), int(r['g2'])) for r in sel})} 个）")
    by = {c["id"]: c for c in cand}
    keep = sorted(r["id"] for r in sel)
    with open(HERE / "scenarios_stress.jsonl", "w", encoding="utf-8") as f:
        for i in keep:
            f.write(json.dumps(by[i], ensure_ascii=False) + "\n")
    with open(HERE / "plans" / "plan_stress.jsonl", "w", encoding="utf-8") as f:
        for i in keep:
            f.write(json.dumps(dict(task="switch", scenario=i, variant=dict(pair=by[i]["switch_pair"][0], stress=True), sample=0,
                                    job_id=f"sw_{i}", seed=SEED), ensure_ascii=False) + "\n")
    df = pd.DataFrame(sel)
    if len(df):
        df["x"] = (df.g1.astype(int) - 1) // 3
        print(df.groupby("x").size().to_string())
        print(df[["w", "h", "prompt_default_worst_ratio", "safe_level"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
