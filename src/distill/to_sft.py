"""原始轨迹 distill/raw/**.json → 过滤打标 → SFT / DPO 数据（Qwen 工具调用格式，HF chat template 直接可用）。

  python distill/to_sft.py                                   # 打标 + 写 data/sft_train.jsonl、data/dpo_pairs.jsonl、data/index.jsonl
  python distill/to_sft.py --reasoning full --max-tokens 32768 --tokenizer Qwen/Qwen3-1.7B --out data_think
  python distill/to_sft.py --compute-ref                     # 先给缺 GA 参照的分组 / 故障状态补跑 NSGA-II（本地，零 API），缓存 distill/ref.json

过滤规则（全部满足才进 SFT，否则记 reject 原因、留作 DPO 负例 / 后续 GRPO）：
  R1 status == ok（非超时、非 API 错误）
  R2 兜底前可行（K = 3 轮内教师自己的方案通过全量 check_constraints）
  R3 最终方案全量复核可行（稳态 + τ_min + 全量瞬态，与正式实验 final_check 同口径）
  R4/R5（只在 --strict 时整条剔除）有格式错误轮 / 触发工具调用上限 12 / 工具报错。默认不剔除，改为逐轮屏蔽损失：
     出错的 assistant 消息（工具调用报错或被上限拒绝、最终 JSON 解析或 Schema 不过、校核不过而收到反馈）标 "train": false，
     只作上下文；其后的反馈与修正照常训练。试点里故障轨迹 69% 触发上限（教师常用 13 次），整条剔除会丢掉大半故障数据。
  R6 质量，按 scoring.py 的方案 A 先比缺水再比 C*：缺水量 ≤ 同状态最小缺水 + d_tol（max(1 m³, 0.2% 总需水)），否则 R6:deficit；
     最小缺水取可行且一致的样本与 GA 方案的较小者（2026-11-02 加 GA：级联下 Flash 独占的状态常照搬剩余轮灌表、多缺水几百 m³）；
     C* 差距 gap = ref − C* ≤ δ（thresholds.json 的 0.02）；ref = 同状态 GA 的 C*（轨迹内 run_optimizer 返回值、ref.json 缓存的 GA 方案 C* 与缺水）
     与同状态缺水达标的最好样本 C* 的较大者。只拿可行且一致的样本定参照：多隔离的方案缺水大、C* 反而可能更高，不能让它抬高参照。
     切换只要求 R1–R5
  R7 转换后长度 ≤ --max-tokens（默认 16384，按 tokenizer 计；没给 tokenizer 时按 UTF-8 字节 / --bytes-per-token 估，默认 2.25 = Qwen3 分词器在故障轨迹上实测，偏保守）
  R8 故障处置的隔离与计划一致性（consistency.py，纯规则）：隔离阀下游支管不得再排入剩余计划、removed_valves 与隔离子树一致、
     隔离阀确有关阀动作、爆管段在隔离子树内、卡开 / 部分卡死阀不作隔离阀、已灌完支管不重排。正式实验的校核按场景真值算稳态，
     看不出这类矛盾（试点里泄漏 6/14、卡死 5/12 的最终方案带着矛盾通过了 R3）。不一致轨迹不进 SFT，作 DPO 负例；
     轨迹里不一致的中间方案（check_constraints 参数、各轮最终 JSON）也标 train = false。
  --repair（故障，零 API）：R8 不过、其余都过的轨迹按 consistency.repair 规则修复，修复方案过一致性版校核（稳态按声明的隔离算 + 全量瞬态）
     后追加一轮"校核未通过 → 修正"：原最终 JSON 屏蔽，反馈是一致性版 check_constraints 的输出，修正 JSON 计损失。
     这一条进 SFT（meta.repaired = true），同时与原最终 JSON 组成同上下文的 DPO 对。教学生"先查一致性、发现矛盾就剔除"。
  每个状态最多保留 --max-per-state 条（默认 2，按 轮数、工具调用数、−C* 排序取前几条）。
DPO 配对：同一状态里 SFT 保留的最好样本作 chosen，R2/R3/R6/R8 不过的样本作 rejected（只配被拒于质量、可行性或一致性的，不配 API 失败）；
  --repair 另出同上下文对（prompt = 修正前的全部消息，chosen = 修正 JSON，rejected = 原最终 JSON）。
故障处置的最终 JSON 按 fault_id、isolate_valves、isolation_actions、removed_valves、remaining_plan 的固定顺序重写：
  学生先写出隔离阀和退出支管，再排剩余计划，退出名单就是显式的自检清单（不另加字段，零额外 token）。

SFT 格式（TRL SFTTrainer / ms-swift / LLaMA-Factory 的 messages + tools 列）：
  {"messages": [{"role":"system"}, {"role":"user"}, {"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":…,"arguments":{…}}}]},
                {"role":"tool","name":…,"content":"…"}, …, {"role":"user","content":"校核未通过…"}, {"role":"assistant","content":"<最终 JSON>"}],
   "tools": [{"type":"function","function":{…}}], "meta": {…}}
  --reasoning none（默认，非思考学生，时延最短）：去掉 reasoning_content；full：保留为 reasoning_content 字段（Qwen3 模板渲染成 <think>）。
  最终 JSON 一律规范化重写（json.dumps 紧凑格式），去掉代码块包裹和多余文字；工具参数解析成 dict（Qwen 模板对 arguments 做 tojson）。
  "train": false 的 assistant 消息不计损失：训练脚本按消息边界把这些 token 的 label 置 -100（TRL 的 assistant_only_loss 不认这个字段，需自带 collator）。
"""
import argparse
import collections
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import consistency as cons                                  # noqa: E402

DELTA = json.load(open(ROOT / "thresholds.json", encoding="utf-8"))["delta"]
DT = {s["id"]: sum(s["demand_m3"].values()) for s in (json.loads(l) for l in open(ROOT / "scenarios.jsonl", encoding="utf-8"))}


def state_key(r):
    v = r["variant"]
    return (r["task"], r["scenario"]["base_id"], v.get("pair", v.get("t", "")))


def load_raw(src):
    return [json.load(open(p, encoding="utf-8")) for p in sorted(src.glob("*/*.json"))]


FR_ORDER = ("fault_id", "isolate_valves", "isolation_actions", "removed_valves", "remaining_plan")


def canonical(d):
    fr = d.get("fault_response")
    if isinstance(fr, dict):
        d = dict(d, fault_response={**{k: fr[k] for k in FR_ORDER if k in fr}, **{k: v for k, v in fr.items() if k not in FR_ORDER}})
    return json.dumps(d, ensure_ascii=False, separators=(",", ":"))


def canonical_final(content):
    import teacher
    return canonical(teacher.parse_decision(content or ""))


def consistent(r, d):
    return r["task"] != "fault_response" or cons.check(d, r["scenario"]["fault"], r["state"]["done"])["consistent"]


def d_tol(r):
    import scoring
    return scoring.d_tol(DT[r["scenario"]["base_id"]])


def good(r):
    """可行且一致、能当参照的样本。"""
    o = r.get("outcome") or {}
    return (r["status"] == "ok" and o.get("final_feasible") and o["feasible_before_fallback"]
            and (o.get("decision") is None or consistent(r, o["decision"])))


FEEDBACK = "校核未通过"


def convert(r, reasoning):
    """原始消息 → Qwen 工具调用格式，并标出不该学的 assistant 消息（train=false）：
    工具调用报错 / 被上限拒绝的那条、紧接着收到校核反馈的那条（格式错或方案不可行）、隔离与计划不一致的方案（R8）。"""
    msgs = r["messages"]
    names, out = {}, []
    for i, m in enumerate(msgs):
        role = m["role"]
        if role in ("system", "user"):
            out.append(dict(role=role, content=m["content"]))
        elif role == "assistant":
            a, bad = dict(role="assistant"), False
            calls = m.get("tool_calls") or []
            j = i + 1
            replies = []
            while j < len(msgs) and msgs[j]["role"] == "tool":
                replies.append(msgs[j]["content"]); j += 1
            if j < len(msgs) and msgs[j]["role"] == "user" and msgs[j]["content"].startswith(FEEDBACK):
                bad = True
            if calls:
                bad |= any(c.startswith('{"error"') for c in replies)
                a["content"] = m.get("content") or ""
                a["tool_calls"] = []
                for c in calls:
                    names[c["id"]] = c["function"]["name"]
                    try:
                        args = json.loads(c["function"]["arguments"] or "{}")
                    except json.JSONDecodeError:                     # 教师参数写坏（已在 replies 里报错，bad 已置）
                        args = c["function"]["arguments"]
                    if c["function"]["name"] == "check_constraints" and isinstance(args, dict) and isinstance(args.get("decision"), dict):
                        bad |= not consistent(r, args["decision"])
                    a["tool_calls"].append(dict(type="function", function=dict(name=c["function"]["name"], arguments=args)))
            else:
                try:
                    a["content"] = canonical_final(m.get("content"))
                    bad |= not consistent(r, json.loads(a["content"]))
                except Exception:
                    a["content"], bad = m.get("content") or "", True
            if reasoning == "full" and m.get("reasoning_content"):
                a["reasoning_content"] = m["reasoning_content"]
            if bad:
                a["train"] = False
            out.append(a)
        elif role == "tool":
            out.append(dict(role="tool", name=names.get(m["tool_call_id"], ""), content=m["content"]))
    return out


class Counter:
    def __init__(self, tokenizer, bpt):
        self.tok, self.bpt = None, bpt
        if tokenizer:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(tokenizer)

    def __call__(self, msgs, tools):
        if self.tok is None:
            return int(len(json.dumps(dict(m=msgs, t=tools), ensure_ascii=False).encode()) / self.bpt)
        return len(self.tok.apply_chat_template(msgs, tools=tools, tokenize=True))


def ga_refs(r):
    """轨迹里 run_optimizer 返回的 GA 方案 (C*, 缺水)：同一状态的 GA 参照，R6 的缺水和 C* 都对它比。"""
    names = {c["id"]: c["function"]["name"] for m in r.get("messages") or [] if m["role"] == "assistant" for c in m.get("tool_calls") or []}
    out = []
    for m in r.get("messages") or []:
        if m["role"] == "tool" and names.get(m.get("tool_call_id")) == "run_optimizer":
            try:
                g = json.loads(m["content"])
                if g.get("feasible") and g.get("score") is not None:
                    out.append((g["score"], g["objectives"]["deficit_m3"]))
            except (json.JSONDecodeError, AttributeError, KeyError, TypeError):
                pass
    return out


def compute_refs(R, refs, path, shard=(0, 1)):
    """缺 GA 参照的分组 / 故障状态补跑优化器，存 GA 方案本身的 C* 与缺水（同 run_optimizer 工具的返回）。
    不用 rule_fault 的整份决策过 final_check：iso_transient 口径下规范最短历时关阀常越限，会把参照整个丢掉，而隔离动作不影响稳态目标。"""
    import distill_gen as g
    import teacher
    import tools
    S = {s["id"]: s for s in (json.loads(l) for l in open(ROOT / "scenarios.jsonl", encoding="utf-8"))}
    have = {state_key(r) for r in R if r["status"] == "ok" and ga_refs(r)}
    todo = {}
    for r in R:
        k = state_key(r)
        if r["task"] != "switch" and r["status"] == "ok" and k not in have and "|".join(map(str, k)) not in refs:
            todo[k] = r
    todo = dict(list(todo.items())[shard[0]::shard[1]])
    print(f"补算 GA 参照 {len(todo)} 个状态（分片 {shard[0]}/{shard[1]}）")
    for i, (k, r) in enumerate(todo.items()):
        job = dict(task=r["task"], scenario=r["scenario"]["base_id"], variant=r["variant"], sample=r["sample"], seed=r["plan_seed"])
        sc = g.scenario_for(job, S)
        ctx, _ = teacher.make_state(r["task"], sc, g.rng_for(job))
        o = tools.run_optimizer(ctx, sc, apply_faults=(r["task"] == "fault_response"))
        refs["|".join(map(str, k))] = dict(score=o["score"], deficit=o["objectives"]["deficit_m3"]) if o["feasible"] else None
        json.dump(refs, open(path, "w", encoding="utf-8"))
        print(f"  {i + 1}/{len(todo)} {k} {refs['|'.join(map(str, k))]}", flush=True)


def fixer():
    """--repair 用：按 consistency.repair 两种口径修复，取过一致性版全量校核且 C* 高的那个。返回 fix(r) -> (决策, 原方案反馈, 复核) 或 None。"""
    import distill_gen as g
    import evalrun
    import teacher
    import tools
    cons.install()
    S = {s["id"]: s for s in (json.loads(l) for l in open(ROOT / "scenarios.jsonl", encoding="utf-8"))}

    def fix(r):
        if r["outcome"]["n_rounds"] > teacher.K_ROUNDS:                  # 修正轮已用完，运行时不会再有一轮
            return None
        job = dict(task=r["task"], scenario=r["scenario"]["base_id"], variant=r["variant"], sample=r["sample"], seed=r["plan_seed"],
                   job_id=r["job_id"])
        sc = g.scenario_for(job, S)
        ctx, st = teacher.make_state(r["task"], sc, g.rng_for(job))
        d0, f = r["outcome"]["decision"], r["scenario"]["fault"]
        ok = []
        for mode in ("plan",) if f["type"] == "burst" else ("plan", "isolation"):
            d = cons.repair(d0, f, st["done"], mode)
            fin = evalrun.final_check(ctx, r["task"], d, st)
            if fin["final_feasible"] and cons.check(d, f, st["done"])["consistent"]:
                ok.append((d, fin))
        if not ok:
            return None
        dmin = min(fin["objectives"]["deficit_m3"] for _, fin in ok)                 # 方案 A：先缺水（容差内）再 C*
        d, fin = max((t for t in ok if t[1]["objectives"]["deficit_m3"] <= dmin + d_tol(r)), key=lambda t: t[1].get("score") or 0)
        return d, tools.check_constraints(ctx, d0, open_before=st.get("g1")), fin
    return fix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", default=[str(HERE / "raw")],
                    help="轨迹目录，可多个（级联：raw_flash raw）；同一 job 的两位教师轨迹都进同一状态，各自过滤")
    ap.add_argument("--out", default="data")
    ap.add_argument("--reasoning", choices=["none", "full"], default="none")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--bytes-per-token", type=float, default=2.25, help="没给 tokenizer 时的估算比（Qwen3 分词器实测：故障轨迹 2.24–2.25，试点混合 2.48；取保守值）")
    ap.add_argument("--max-per-state", type=int, default=2)
    ap.add_argument("--max-tool-errors", type=int, default=0)
    ap.add_argument("--compute-ref", action="store_true")
    ap.add_argument("--ref-shard", default=None, help="i/k：只补算第 i 片 GA 参照、写 ref.s<i>.json 后退出（并行起 k 个进程，GA 走全机槽锁）")
    ap.add_argument("--strict", action="store_true", help="格式错误轮 / 触发上限 / 工具报错的轨迹整条剔除（默认逐轮屏蔽）")
    ap.add_argument("--repair", action="store_true", help="R8 不过的故障轨迹按规则修复并追加一轮修正（本地仿真复核，零 API）")
    a = ap.parse_args()
    fix = fixer() if a.repair else None
    R = [r for s in a.src for r in load_raw(pathlib.Path(s))]
    ref_path = HERE / "ref.json"
    refs = {}
    for f in [ref_path] + sorted(HERE.glob("ref.s*.json")):                # 并行分片写的各自文件，读时合并
        if f.exists():
            refs.update(json.load(open(f, encoding="utf-8")))
    if a.ref_shard:
        si, sk = map(int, a.ref_shard.split("/"))
        return compute_refs(R, refs, HERE / f"ref.s{si}.json", (si, sk))
    if a.compute_ref:
        compute_refs(R, refs, ref_path)

    by_state = collections.defaultdict(list)
    for r in R:
        by_state[state_key(r)].append(r)
    ref, dmin = {}, {}
    for k, rs in by_state.items():
        ga = [t for r in rs if r["status"] == "ok" for t in ga_refs(r)]
        if refs.get("|".join(map(str, k))) is not None:
            ga.append(tuple(refs["|".join(map(str, k))].values()))
        ds = [r["outcome"]["objectives"]["deficit_m3"] for r in rs if good(r) and r["outcome"].get("objectives")] + [d for _, d in ga]
        dmin[k] = min(ds) if ds else None                            # 缺水参照含 GA：Flash 常照搬剩余轮灌表、多缺水几百 m³，同状态样本挡不住
        cand = [s for s, _ in ga]
        cand += [r["outcome"]["score"] for r in rs if good(r) and r["outcome"].get("score") is not None
                 and r["outcome"]["objectives"]["deficit_m3"] <= dmin[k] + d_tol(r)]
        ref[k] = max(cand) if cand else None

    count = Counter(a.tokenizer, a.bytes_per_token)
    index, keep, repaired_pairs, by_rec = [], collections.defaultdict(list), [], {}
    for r in R:
        why, o, n_tok, conv, gap, rep = [], r.get("outcome") or {}, None, None, None, None
        if r["status"] != "ok":
            why.append(f"R1:{r['status']}")
        else:
            if not o["feasible_before_fallback"]:
                why.append("R2:infeasible_before_fallback")
            if not o.get("final_feasible"):
                why.append("R3:final_infeasible")
            if a.strict and o["format_errors"]:
                why.append("R4:format_error")
            if a.strict and (o["tool_cap_hit"] or o["tool_errors"] > a.max_tool_errors):
                why.append("R5:tool_cap_or_error")
            if o.get("decision") and not consistent(r, o["decision"]):
                c = cons.check(o["decision"], r["scenario"]["fault"], r["state"]["done"])
                why.append("R8:" + "+".join(sorted({i["code"] for i in c["issues"] if i["code"] in cons.HARD})))
            score, dfc = o.get("score"), (o.get("objectives") or {}).get("deficit_m3")
            conv = convert(r, a.reasoning)
            if fix and why and all(w.startswith("R8") for w in why) and conv[-1]["role"] == "assistant":
                rep = fix(r)
                if rep:
                    d1, fb, fin = rep
                    repaired_pairs.append((r, conv[:-1], conv[-1], d1))
                    conv = conv + [dict(role="user", content="校核未通过，请根据以下结构化反馈修改方案，仍只输出 JSON：\n" +
                                        json.dumps(fb, ensure_ascii=False, default=float)),
                                   dict(role="assistant", content=canonical(d1))]
                    why, score, dfc = [], fin.get("score"), fin["objectives"]["deficit_m3"]
            if r["task"] != "switch" and dfc is not None and dmin[state_key(r)] is not None and dfc > dmin[state_key(r)] + d_tol(r):
                why.append("R6:deficit")
            if r["task"] != "switch" and score is not None and ref[state_key(r)] is not None:
                gap = round(ref[state_key(r)] - score, 4)
                if gap > DELTA:
                    why.append("R6:gap")
            n_tok = count(conv, [dict(type="function", function=s) for s in r["tools"]])
            if n_tok > a.max_tokens:
                why.append("R7:too_long")
        masked = sum(1 for m in conv if m.get("train") is False) if conv else None
        rec = dict(job_id=r["job_id"], teacher=r.get("served_model") or r["teacher"]["model"], task=r["task"], state=list(state_key(r)),
                   status=r["status"], keep=not why, reject=why,
                   repaired=bool(rep), gap=gap, score=score if r["status"] == "ok" else None, n_rounds=o.get("n_rounds"), tool_calls=o.get("tool_calls"), n_tokens=n_tok,
                   cost_cny=(r.get("cost") or {}).get("cost_cny"), fault=(r["scenario"].get("fault") or {}).get("type"),
                   masked_turns=masked, cap_hit=o.get("tool_cap_hit"), tool_errors=o.get("tool_errors"), format_errors=o.get("format_errors"))
        index.append(rec)
        by_rec[id(r)] = rec
        if not why:
            keep[state_key(r)].append((rec, r, conv))

    out = HERE / a.out
    out.mkdir(exist_ok=True)
    n_sft = n_dpo = 0
    with open(out / "sft_train.jsonl", "w", encoding="utf-8") as fs, open(out / "dpo_pairs.jsonl", "w", encoding="utf-8") as fd:
        for k, ks in keep.items():
            ks.sort(key=lambda t: (t[0]["n_rounds"], t[0]["tool_calls"], -(t[0]["score"] or 0)))
            for rec, r, conv in ks[:a.max_per_state]:
                tools_ = [dict(type="function", function=s) for s in r["tools"]]
                fs.write(json.dumps(dict(messages=conv, tools=tools_, meta=dict(job_id=r["job_id"], task=r["task"], scenario=r["scenario"]["id"],
                                                                                 fault=(r["scenario"].get("fault") or {}).get("type"),
                                                                                 n_tokens=rec["n_tokens"], gap=rec["gap"], repaired=rec["repaired"],
                                                                                 teacher=rec["teacher"])),
                                     ensure_ascii=False) + "\n")
                n_sft += 1
            best = ks[0][1]
            chosen = ks[0][2] if ks[0][0]["repaired"] else convert(best, a.reasoning)
            for r in by_state[k]:
                rj = by_rec[id(r)]                                   # 级联下同一 job_id 有两位教师的轨迹
                if rj["keep"] or not any(w.startswith(("R2", "R3", "R6", "R8")) for w in rj["reject"]):
                    continue
                try:
                    rejected = convert(r, a.reasoning)
                except Exception:                                    # 拒绝样本含解析不了的最终 JSON：保留原文
                    rejected = [dict(role=m["role"], content=m.get("content") or "") for m in r["messages"]]
                fd.write(json.dumps(dict(prompt=chosen[:2], chosen=chosen[2:], rejected=rejected[2:], tools=[dict(type="function", function=s) for s in best["tools"]],
                                         meta=dict(state=list(k), chosen=best["job_id"], rejected=r["job_id"], reject=rj["reject"],
                                                   chosen_teacher=ks[0][0]["teacher"], rejected_teacher=rj["teacher"])), ensure_ascii=False) + "\n")
                n_dpo += 1
        for r, ctx_msgs, bad, d1 in repaired_pairs:                       # 同上下文：一致的修正方案 vs 原来不一致的方案
            fd.write(json.dumps(dict(prompt=ctx_msgs, chosen=[dict(role="assistant", content=canonical(d1))],
                                     rejected=[{k: v for k, v in bad.items() if k != "train"}],
                                     tools=[dict(type="function", function=s) for s in r["tools"]],
                                     meta=dict(state=list(state_key(r)), chosen=r["job_id"] + "~repair", rejected=r["job_id"], reject=["R8"])),
                                ensure_ascii=False) + "\n")
            n_dpo += 1
    with open(out / "index.jsonl", "w", encoding="utf-8") as f:
        for rec in index:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"原始 {len(R)} 条，SFT {n_sft} 条，DPO 对 {n_dpo}，状态 {len(by_state)}")
    for t in ("switch", "rotation_plan", "fault_response"):
        rs = [i for i in index if i["task"] == t]
        if rs:
            rej = collections.Counter(w.split(":")[0] for i in rs for w in i["reject"])
            toks = sorted(i["n_tokens"] for i in rs if i["n_tokens"])
            kept = [i for i in rs if i["keep"]]
            print(f"  {t:<15} {len(rs)} 条，通过 {len(kept)}（修复 {sum(1 for i in kept if i['repaired'])}），拒绝 {dict(rej)}，"
                  f"通过者含屏蔽轮 {sum(1 for i in kept if i['masked_turns'])} 条，"
                  f"长度 中位 {toks[len(toks) // 2] if toks else '-'} / P95 {toks[int(len(toks) * .95)] if toks else '-'} token")


if __name__ == "__main__":
    main()
