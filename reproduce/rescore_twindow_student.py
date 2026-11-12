"""学生（本机 repro30×10、三档精度、CPU 探针；云上 val250 的 v2 s312 与 v3 = DPO a s85）与教师 repro 的时间窗容差重判（只读）。
输入：extract_checks.py 压出的 jsonl；教师 results/api/repro_ds.jsonl。
口径（与 data.apply_twin_tol 一致）：
  兜底前：最后一次 check_constraints 不可行，且唯一失败的校核是稳态、稳态唯一越限项是 T_window ≤ TOL → 改判可行。
  兜底后：最终复核不可行，且稳态唯一越限项是 T_window ≤ TOL、无瞬态 / τ 越限、无复核报错 → 改判可行。
  达标 = 兜底前可行 + 隔离与计划一致（故障）+ 缺水 ≤ GA + d_tol + C* 差距 ≤ δ（切换只看可行），同 eval_student.row_metrics；
  改判的记录用最后一次校核里的目标值重算 C*。出错（上下文溢出）按失败计。
  python rescore_twindow_student.py <src 目录> <输出 md> <checks jsonl> [...]"""
import collections, json, pathlib, sys

STAGE, OUT, FILES = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
sys.path.insert(0, str(STAGE))
import scoring                                                        # noqa: E402

TOL = 0.03
DELTA = json.load(open(STAGE / "thresholds.json", encoding="utf-8"))["delta"]
DT = {s["id"]: sum(s["demand_m3"].values()) for s in (json.loads(l) for l in open(STAGE / "scenarios.jsonl", encoding="utf-8"))}
TASKS = ("switch", "rotation_plan", "fault_response")
TN = dict(switch="切换", rotation_plan="分组", fault_response="故障")


def tw_only(v, tol=TOL):
    return bool(v) and set(v) == {"T_window"} and v["T_window"] <= tol


def judge(r):
    ok = r["status"] == "ok"
    bf = ok and bool(r["bf"])
    bf_tol = bf or (ok and r.get("pre_feasible") is False and not r.get("pre_format_error")
                    and r.get("pre_failed_checks") == ["steady"] and tw_only(r.get("pre_steady_viol")))
    fin = ok and bool(r["final_feasible"])
    fin_tol = fin or (ok and r["final_feasible"] is False and tw_only(r.get("final_viol")) and not r.get("final_trans_n")
                      and not r.get("final_tau_n") and not r.get("final_check_error"))
    cons = r["task"] != "fault_response" or (ok and (bool(r.get("consistent")) or (bf_tol and not bf and "consistency" not in (r.get("pre_failed_checks") or []))))

    def good_with(obj, score):
        if r["task"] == "switch":
            return True
        ref = r.get("ref") or {}
        if not obj or score is None or ref.get("deficit") is None or ref.get("score") is None:
            return False
        return obj["deficit_m3"] - ref["deficit"] <= scoring.d_tol(DT[r["base_id"]]) and ref["score"] - score <= DELTA
    good = bf and cons and good_with(r.get("final_obj"), r.get("score"))
    if bf:
        good_tol = good
    elif bf_tol:
        obj = r.get("pre_obj")
        good_tol = cons and good_with(obj, scoring.closeness(obj) if obj else None)
    else:
        good_tol = False
    # 兜底前失败原因（容差重判之后剩下的）
    if bf_tol:
        why = None
    elif not ok:
        why = "上下文溢出" if r.get("ctx_err") else "其他出错"
    elif r.get("pre_format_error"):
        why = "工具上限后未给出可解析方案" if r.get("cap") else "未给出可解析方案"
    else:
        fc = set(r.get("pre_failed_checks") or [])
        sv = set((r.get("pre_steady_viol") or {}).keys())
        if "consistency" in fc or "isolated_valve_assigned" in sv:
            why = "隔离与计划不一致"
        elif "isolation" in fc:
            why = "隔离阀未关"
        elif "transient" in fc:
            why = "瞬态越限"
        elif sv - {"T_window"}:
            why = "其他稳态约束（" + "、".join(sorted(sv - {"T_window"})) + "）"
        elif "T_window" in sv:
            why = f"时间窗越限 > {TOL} h"
        elif "tau_min" in fc:
            why = "最短启闭时间"
        else:
            why = "其他（" + ",".join(sorted(fc)) + "）"
    if r.get("cap") and why:
        why += "［工具撑满］"
    return dict(bf=bf, bf_tol=bf_tol, good=good, good_tol=good_tol, fin=fin, fin_tol=fin_tol, why=why)


GROUPS = [
    ("本机 repro30×10（test，Q8_0 + f16 KV，部署配置）", ["v3_s00085_q8_repro"]),
    ("本机精度对比 Q8_0（val_ckpt 60，KV q8_0）", ["v3_s00085_q8_kvq8"]),
    ("本机精度对比 bf16（val_ckpt 60，KV q8_0）", ["v3_s00085_bf16_kvq8"]),
    ("本机精度对比 Q4_K_M（val_ckpt 60，KV q8_0）", ["v3_s00085_q4km_kvq8"]),
    ("本机纯 CPU 探针（9 条）", ["v3_s00085_q8_cpu"]),
    ("云上 v2 s312 val250（val_ckpt 60 + val_big 190）", ["v2_s00312_q8_cloud", "v2_s00312_q8_big_cloud"]),
    ("云上 v3 = DPO a s85 val250（val_ckpt 60 + val_big 190）", ["dpo_a_s00085_q8_cloud", "dpo_a_s00085_q8_big_cloud"]),
    ("云上 v2 s312 只看 val_ckpt 60", ["v2_s00312_q8_cloud"]),
    ("云上 v3 只看 val_ckpt 60", ["dpo_a_s00085_q8_cloud"]),
]

rows = [json.loads(l) for f in FILES for l in open(f, encoding="utf-8")]
by = collections.defaultdict(list)
for r in rows:
    r.update(judge(r))
    by[r["tag"]].append(r)

L = [f"# 时间窗容差重判：学生与教师（容差 {TOL} h；只读已有结果）\n",
     "口径：兜底前以最后一次 check_constraints 为准，唯一失败校核为稳态且稳态唯一越限项为 T_window ≤ 容差则改判可行；"
     "兜底后最终复核同理（另须无瞬态 / τ 越限、无复核报错）。达标 = 兜底前可行 + 隔离一致（故障）+ 缺水 ≤ GA + d_tol + C* 差距 ≤ δ = "
     f"{DELTA}，改判记录用其最后一次校核的目标值重算 C*。出错（上下文溢出）按失败计。原判 → 重判。\n"]
for name, tags in GROUPS:
    rs = [r for t in tags for r in by.get(t, [])]
    if not rs:
        L.append(f"\n## {name}\n\n（暂无数据）\n")
        continue
    L.append(f"\n## {name}\n\n| 任务 | n | 兜底前可行 | 达标 | 兜底后最终可行 | 改判条数（兜底前 / 最终） |\n|---|---|---|---|---|---|")
    for task in TASKS + ("合计",):
        s = rs if task == "合计" else [r for r in rs if r["task"] == task]
        if not s:
            continue
        c = lambda k: sum(r[k] for r in s)
        L.append(f"| {TN.get(task, task)} | {len(s)} | {c('bf')} → {c('bf_tol')} | {c('good')} → {c('good_tol')} | {c('fin')} → {c('fin_tol')} | "
                 f"{c('bf_tol') - c('bf')} / {c('fin_tol') - c('fin')} |")
    why = collections.Counter((r["task"], r["why"]) for r in rs if r["why"])
    if why:
        L.append("\n重判后仍兜底前不可行的原因：" + "；".join(f"{TN[t]} {w} {n}" for (t, w), n in sorted(why.items(), key=lambda x: (TASKS.index(x[0][0]), -x[1]))))
    tw = sorted(r["pre_steady_viol"]["T_window"] for r in rs if r["bf_tol"] and not r["bf"])
    if tw:
        L.append(f"\n被改判的兜底前越限量（h）：最小 {tw[0]}，最大 {tw[-1]}，中位 {tw[len(tw) // 2]}。")
    ft = sorted(r["final_viol"]["T_window"] for r in rs if r["fin_tol"] and not r["fin"])
    if ft:
        L.append(f"被改判的兜底后越限量（h）：最小 {ft[0]}，最大 {ft[-1]}。")


COMPACT = [
    ("1.7B v1 末轮 s295 Q8_0", "v1_s00295_q8_cloud"), ("1.7B v2 末轮 s312 Q8_0", "v2_s00312_q8_cloud"),
    ("1.7B v2 末轮 bf16", "v2_s00312_bf16kv16_cloud"), ("1.7B v2 末轮 Q4_K_M", "v2_s00312_q4km_cloud"),
    ("4B 末轮 s295（v1 数据）", "q4b_s00295_q8_cloud"), ("8B 末轮 s295（v1 数据）", "q8b_s00295_q8_cloud"),
    ("v3 = DPO s85 Q8_0", "dpo_a_s00085_q8_cloud"),
    ("v3 本机 Q8_0（KV q8_0）", "v3_s00085_q8_kvq8"), ("v3 本机 bf16（KV q8_0）", "v3_s00085_bf16_kvq8"),
    ("v3 本机 Q4_K_M（KV q8_0）", "v3_s00085_q4km_kvq8"),
]
L.append("\n## 云上规模与量化对比，及本机三档精度（val_ckpt 60，单次运行；兜底前可行 / 达标 / 兜底后最终，原判 → 重判）\n\n"
         "| 模型 | 全部 60 | 故障 30 | 分组 10 | 切换 20 | 出错 |\n|---|---|---|---|---|---|")
for lab, t in COMPACT:
    rs = by.get(t, [])
    if not rs:
        continue

    def cell(s):
        c = lambda k: sum(r[k] for r in s)
        return f"{c('bf')}/{c('good')}/{c('fin')} → {c('bf_tol')}/{c('good_tol')}/{c('fin_tol')}"
    L.append(f"| {lab} | {cell(rs)} | " + " | ".join(cell([r for r in rs if r['task'] == k]) for k in ("fault_response", "rotation_plan", "switch"))
             + f" | {sum(r['status'] != 'ok' for r in rs)} |")


V2 = {(r["tag"].replace("v2_s00312_q8_", ""), r["job_id"]): r for t in ("v2_s00312_q8_cloud", "v2_s00312_q8_big_cloud") for r in by.get(t, [])}
V3 = {(r["tag"].replace("dpo_a_s00085_q8_", ""), r["job_id"]): r for t in ("dpo_a_s00085_q8_cloud", "dpo_a_s00085_q8_big_cloud") for r in by.get(t, [])}
keys = sorted(set(V2) & set(V3))
if keys:
    L.append(f"\n## 云上 val250 同一批场景上 v3 与 v2 的计数（n = {len(keys)}）\n\n| 指标 | 口径 | v3 | v2 | 仅 v3 / 仅 v2 |\n|---|---|---|---|---|")
    for k, lab in (("bf", "兜底前可行"), ("good", "达标"), ("fin", "兜底后最终可行")):
        for suf, kou in (("", "原判"), ("_tol", "容差重判")):
            a = [V3[x][k + suf] for x in keys]; b = [V2[x][k + suf] for x in keys]
            L.append(f"| {lab} | {kou} | {sum(a)} | {sum(b)} | {sum(x and not y for x, y in zip(a, b))} / {sum(y and not x for x, y in zip(a, b))} |")

# 教师 repro（DeepSeek-V4-Pro，原工作流）
T = [json.loads(l) for l in open(STAGE.parent / "results" / "api" / "repro_ds.jsonl", encoding="utf-8")]
L.append("\n## 教师 repro30×10（DeepSeek-V4-Pro，test，同一批状态）\n\n| 任务 | n | 兜底前可行 | 兜底后最终可行 |\n|---|---|---|---|")
for task in TASKS:
    s = [r for r in T if r["task"] == task and r.get("status", "ok") == "ok"]
    bf = sum(bool(r.get("feasible_before_fallback")) for r in s)
    fin = sum(bool(r.get("final_feasible")) for r in s)
    fin_tol = sum(bool(r.get("final_feasible")) or (tw_only(r.get("violations")) and not r.get("transient_violations") and not r.get("tau_violations")
                                                     and not r.get("check_error")) for r in s)
    L.append(f"| {TN[task]} | {len(s)} | {bf}（无法重判） | {fin} → {fin_tol} |")
L.append("\n教师行只存了最终（兜底后）方案的越限，没有存模型自己最后提交方案的校核，兜底前的 9 例故障失败（F00197 r7、F00205 r6/r8、F00502 r4、"
         "F00278 r4、F00964 r1/r8/r9、F00133 r9）无法判断是否属同类舍入；其中 5 例兜底后的越限是 T_window 0.000–0.002 h。")
pathlib.Path(OUT).write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
