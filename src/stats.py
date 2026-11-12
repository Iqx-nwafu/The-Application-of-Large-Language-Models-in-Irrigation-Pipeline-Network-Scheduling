"""A9 统计脚本：汇总 evalrun 的记录并做检验。

python stats.py results/pre_*.jsonl [--out results/summary]

按 (task, maker, model, ablation) 汇总：
  n；timeout（超时未完成条数，status 字段）；兜底前可行率、实时达标率、最终可行率（各附 Wilson 95% 区间）；兜底率；最终越限率；
  时延 wall_s、费用、token 总数、C*、与优化器差距、首个可行方案时延的中位数与四分位距。
检验（同一 task 内按场景配对，指标 = C*（分组/故障）或 action_span_s（切换）以及 wall_s）：
  决策器 ≥ 3 个时 Friedman + Nemenyi 事后（CD = q_α·sqrt(k(k+1)/6n)）；两两 Wilcoxon 符号秩。
"""
import argparse
import glob
import json
import pathlib
import numpy as np
import pandas as pd
from scipy import stats

HERE = pathlib.Path(__file__).parent


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(c - h, 3), round(c + h, 3))


def load(patterns):
    rows = []
    for pat in patterns:
        for fn in glob.glob(pat):
            for l in open(fn, encoding="utf-8"):
                r = json.loads(l)
                lat, tok = r.get("latency") or {}, r.get("tokens") or {}
                rows.append(dict(task=r["task"], maker=r["maker"], model=r.get("model") or "-", ablation=r.get("ablation") or "-",
                                 scenario=r["scenario"], repeat=r["repeat"], status=r.get("status", "ok"),
                                 feasible_before=r.get("feasible_before_fallback"), final_feasible=r.get("final_feasible"),
                                 fallback=r.get("fallback_used") is not None, in_window=r.get("in_window"),
                                 wall_s=lat.get("wall_s"), first_feasible_s=lat.get("first_feasible_s"),
                                 cost=r.get("cost_cny"), tokens=(tok.get("prompt", 0) + tok.get("completion", 0)) if tok else None,
                                 score=r.get("score"), gap=r.get("gap_to_optimizer"), span=r.get("action_span_s"),
                                 p_max_ratio=(r.get("transient") or {}).get("p_max_ratio"),
                                 transient_viol=r.get("transient_violations", 0), tool_calls=r.get("tool_calls")))
    return pd.DataFrame(rows)


def med_iqr(x):
    x = pd.Series(x).dropna()
    return f"{x.median():.3g} [{x.quantile(.25):.3g}, {x.quantile(.75):.3g}]" if len(x) else "-"


def summary(df):
    out = []
    for key, g in df.groupby(["task", "maker", "model", "ablation"]):
        n = len(g)
        ok = g[g.status == "ok"]                       # 超时未完成的记录：只计入 n、timeout 与实时达标率（记未达标），其余指标不计
        fb = ok.feasible_before.dropna()
        row = dict(task=key[0], maker=key[1], model=key[2], ablation=key[3], n=n, timeout=int((g.status != "ok").sum()),
                   feasible_before=f"{fb.mean():.2f} {wilson(int(fb.sum()), len(fb))}" if len(fb) else "-",
                   fallback_rate=round(g.fallback.mean(), 2),
                   final_feasible=f"{ok.final_feasible.mean():.2f} {wilson(int(ok.final_feasible.sum()), len(ok))}" if len(ok) else "-",
                   transient_viol_rate=round((g.transient_viol > 0).mean(), 2),
                   in_window=f"{g.in_window.astype(bool).mean():.2f} {wilson(int(g.in_window.astype(bool).sum()), n)}",
                   wall_s=med_iqr(g.wall_s), first_feasible_s=med_iqr(g.first_feasible_s), cost_cny=med_iqr(g.cost),
                   cost_total_cny=round(float(g.cost.fillna(0).sum()), 3),
                   tokens=med_iqr(g.tokens), tool_calls=med_iqr(g.tool_calls),
                   score=med_iqr(g.score), gap_to_opt=med_iqr(g.gap), action_span_s=med_iqr(g.span), p_max_ratio=med_iqr(g.p_max_ratio))
        out.append(row)
    return pd.DataFrame(out)


def tests(df, metric):
    """同一 task 内，按 (scenario, repeat) 配对比较各决策器的 metric。"""
    res = []
    for task, g in df.groupby("task"):
        g = g.assign(who=g.maker + "/" + g.model + "/" + g.ablation)
        piv = g.pivot_table(index=["scenario", "repeat"], columns="who", values=metric).dropna()
        k, n = piv.shape[1], len(piv)
        if k < 2 or n < 2:
            continue
        line = dict(task=task, metric=metric, n_pairs=n, makers=list(piv.columns))
        if k >= 3:
            st, p = stats.friedmanchisquare(*[piv[c].values for c in piv.columns])
            ranks = piv.rank(axis=1).mean()
            q = stats.studentized_range.ppf(0.95, k, 1e6) / np.sqrt(2)
            cd = q * np.sqrt(k * (k + 1) / (6 * n))
            line.update(friedman_p=round(float(p), 4), mean_rank=ranks.round(2).to_dict(), nemenyi_CD=round(float(cd), 3),
                        nemenyi_signif=[(a, b) for i, a in enumerate(piv.columns) for b in list(piv.columns)[i + 1:]
                                        if abs(ranks[a] - ranks[b]) > cd])
        pw = {}
        cols = list(piv.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                d = piv[a] - piv[b]
                if (d != 0).sum() >= 2:
                    pw[f"{a} vs {b}"] = round(float(stats.wilcoxon(piv[a], piv[b]).pvalue), 4)
        line["wilcoxon_p"] = pw
        res.append(line)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=str(HERE / "results" / "summary"))
    a = ap.parse_args()
    df = load(a.files)
    if df.empty:
        print("没有记录"); return
    s = summary(df)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 40)
    print(s.to_string(index=False))
    llm = df[df.model != "-"]
    if len(llm):
        print("\n各模型累计费用（元，按调用时段单价）:", llm.groupby("model").cost.sum().round(3).to_dict(),
              "调用次数:", llm.groupby("model").size().to_dict())
    s.to_csv(a.out + ".csv", index=False)
    t = tests(df, "score") + tests(df, "span") + tests(df, "wall_s")
    json.dump(t, open(a.out + "_tests.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    for line in t:
        print(f"\n[{line['task']} / {line['metric']}] n={line['n_pairs']}", "Friedman p=%s" % line.get("friedman_p"),
              "平均秩", line.get("mean_rank"), "CD", line.get("nemenyi_CD"))
        print("  显著对（Nemenyi）:", line.get("nemenyi_signif"), "\n  Wilcoxon:", line["wilcoxon_p"])
    print("→", a.out + ".csv /", a.out + "_tests.json")


if __name__ == "__main__":
    main()
