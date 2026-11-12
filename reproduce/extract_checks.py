"""把学生评测记录里"兜底前最后一次校核"与"兜底后最终复核"的越限明细压成 jsonl，供时间窗容差重判（只读，标准库）。
  python extract_checks.py <输出 jsonl> <results/student/<tag>.jsonl.gz> [...]
每个 .jsonl.gz 一行一条评测记录（eval_student.py 写的 distill-traj/1 记录，_path 为原相对路径）。"""
import gzip, json, os, sys

out, dirs = sys.argv[1], sys.argv[2:]
MAX_TOKENS = 4096


def pre_summary(rounds):
    if not rounds:
        return dict(pre_n_rounds=0)
    chk = rounds[-1].get("check") or {}
    c = chk.get("checks") or {}
    st = c.get("steady") or {}
    failed = sorted(k for k, v in c.items() if not v.get("feasible", True))
    return dict(pre_n_rounds=len(rounds), pre_feasible=chk.get("feasible"), pre_format_error=chk.get("format_error"),
                pre_failed_checks=failed, pre_steady_viol=st.get("violations"), pre_obj=st.get("objectives"),
                pre_tau_n=len((c.get("tau_min") or {}).get("violations") or []), pre_trans_n=(c.get("transient") or {}).get("n_violations"),
                pre_iso_missing=(c.get("isolation") or {}).get("missing_isolation_valves"),
                pre_cons_codes=[i.get("code") for i in ((c.get("consistency") or {}).get("issues") or [])])


n = 0
with open(out, "w", encoding="utf-8") as f:
    for d in dirs:
        tag = os.path.basename(d).replace(".jsonl.gz", "")
        for line in gzip.open(d, "rt", encoding="utf-8"):
            r = json.loads(line)
            o = r.get("outcome") or {}
            sc = r.get("scenario") or {}
            usage = [m.get("_usage") or {} for m in (r.get("messages") or r.get("partial_messages") or []) if m.get("role") == "assistant"]
            err = r.get("error") or ""
            ctx_full = any(u.get("finish_reason") == "length" and (u.get("completion_tokens") or 0) < MAX_TOKENS for u in usage)
            cons = o.get("consistency") or {}
            L = r.get("latency") or {}
            row = dict(tag=tag, job_id=r.get("job_id"), task=r.get("task"),
                       base_id=sc.get("base_id") if isinstance(sc, dict) else sc, fault=((sc.get("fault") or {}).get("type") if isinstance(sc, dict) else None),
                       sample=r.get("sample"), status=r.get("status"), error=err[:160],
                       ctx_err=(r.get("status") != "ok" and "context" in err.lower()) or ctx_full,
                       cap=bool(o.get("tool_cap_hit")), tool_calls=o.get("tool_calls"), format_errors=o.get("format_errors"),
                       bf=o.get("feasible_before_fallback"), final_feasible=o.get("final_feasible"), fallback=o.get("fallback_used"),
                       final_viol=o.get("violations"), final_trans_n=o.get("transient_violations"), final_tau_n=o.get("tau_violations"),
                       final_check_error=o.get("check_error"), consistent=cons.get("consistent") if cons else None,
                       cons_codes=[i.get("code") for i in (cons.get("issues") or [])] if cons else None,
                       final_obj=o.get("objectives"), score=o.get("score"), ref=r.get("ref"),
                       in_window=L.get("in_window"), wall_s=L.get("wall_s"), **pre_summary(r.get("rounds") or []))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
print("rows", n, "->", out)
