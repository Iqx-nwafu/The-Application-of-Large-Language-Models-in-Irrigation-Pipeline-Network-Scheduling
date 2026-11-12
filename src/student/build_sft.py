"""训练集生成 + 长度统计：从不限长的 to_sft 候选池按真实 Qwen3 token 数定上下文上限，每个状态取前 per_state 条。

先在 irrig 环境出候选池（R1–R6、R8 照常过滤；长度不卡、每状态全留，排序在本脚本里做）：
  python distill/to_sft.py --src distill/raw_flash distill/raw --repair --max-tokens 100000000 --max-per-state 99 --out <work>/pool_<名>
再在 student 环境统计 / 写训练集：
  python src\\student\\build_sft.py <work>/pool_<名> [--lens 16384 24576 32768] [--write 32768 <out.jsonl>]
--max-steady N（v1 起，作者 2026-11-05 定"方案 A"）：故障轨迹里 simulate_steady 调用超过 N 次的整条不要——v0 学生学到教师的反复仿真
却学不到何时停（教师的取舍在被去掉的思考内容里），在 val 上循环到撑满上下文；只留能收敛的短路径。
每状态的取舍与 to_sft 相同（按 轮数、工具调用数、−C* 排序取前 2 条），只是 R7 换成真实 token 数：to_sft 无分词器时按 2.25 字节/token 估，
偏保守，会多丢一些其实放得下的样本。每轮训练时长按本机冒烟实测吞吐分段插值估算（1.7B bf16；4B 4-bit 按同长度 0.43 倍）。
"""
import argparse
import bisect
import collections
import json
import pathlib
import statistics as st
from sft_data import Encoder

TPS_17B = [(4096, 1200), (8192, 1200), (16384, 875), (24576, 868), (32768, 750)]   # 11-02/11-03 冒烟实测 tok/s（offload；32k 为 kv 头分块）
R4B = 0.43                                                                          # 4B 4-bit / 1.7B：24k 384/868、32k 314/750


def tps(n, table=TPS_17B):
    xs = [x for x, _ in table]
    i = min(max(bisect.bisect_left(xs, n), 1), len(xs) - 1)
    (x0, y0), (x1, y1) = table[i - 1], table[i]
    return y0 + (y1 - y0) * (min(max(n, x0), x1) - x0) / (x1 - x0)


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool")
    ap.add_argument("--lens", type=int, nargs="+", default=[16384, 24576, 32768])
    ap.add_argument("--per-state", type=int, default=2)
    ap.add_argument("--model", default="models/Qwen3-1.7B")
    ap.add_argument("--write", nargs=2, metavar=("L", "OUT"), default=None)
    ap.add_argument("--json", default=None, help="统计结果另存 JSON")
    ap.add_argument("--max-steady", type=int, default=None, help="故障轨迹 simulate_steady 调用次数上限（超过的整条剔除）")
    a = ap.parse_args()
    pool = pathlib.Path(a.pool)
    idx = {(r["job_id"], r["teacher"]): r for r in map(json.loads, open(pool / "index.jsonl", encoding="utf-8"))}
    enc = Encoder(a.model)
    rows, n_long = [], collections.Counter()
    for line in open(pool / "sft_train.jsonl", encoding="utf-8"):
        r = json.loads(line)
        m = r["meta"]
        ids, mask = enc(r)
        i = idx[(m["job_id"], m["teacher"])]
        key = ("fault_" + m["fault"]) if m["task"] == "fault_response" else m["task"]
        steady = sum(c["function"]["name"] == "simulate_steady" for x in r["messages"] if x["role"] == "assistant" for c in x.get("tool_calls") or [])
        if a.max_steady is not None and m["task"] == "fault_response" and steady > a.max_steady:
            n_long[key] += 1
            continue
        rows.append(dict(line=line, state=tuple(i["state"]), key=key, n=len(ids), lab=sum(mask[1:]),
                         sort=(i["n_rounds"], i["tool_calls"], -(i["score"] or 0))))
    by_state = collections.defaultdict(list)
    for r in rows:
        by_state[r["state"]].append(r)
    for rs in by_state.values():
        rs.sort(key=lambda r: r["sort"])

    def select(L):
        return [r for rs in by_state.values() for r in [x for x in rs if x["n"] <= L][:a.per_state]]

    keys = sorted({r["key"] for r in rows})
    if a.max_steady is not None:
        print(f"稳态仿真 > {a.max_steady} 次剔除：{dict(n_long)}")
    print(f"候选池 {len(rows)} 条、{len(by_state)} 个状态；真实 token 中位 {int(st.median(r['n'] for r in rows))}、P95 {q([r['n'] for r in rows], .95)}、最大 {max(r['n'] for r in rows)}")
    print("\n各类候选长度（真实 token）：")
    for k in keys:
        v = [r["n"] for r in rows if r["key"] == k]
        print(f"  {k:<16} {len(v):>5} 条  中位 {int(st.median(v)):>6}  P90 {q(v, .9):>6}  P95 {q(v, .95):>6}  最大 {max(v):>6}  "
              + "  ".join(f"≤{L // 1024}k {sum(x <= L for x in v) / len(v):.0%}" for L in a.lens))
    out = {}
    print(f"\n按上限取训练集（每状态 ≤ {a.per_state} 条）：")
    for L in a.lens:
        sel = select(L)
        c = collections.Counter(r["key"] for r in sel)
        states = {r["state"] for r in sel}
        tok, lab = sum(r["n"] for r in sel), sum(r["lab"] for r in sel)
        h17 = sum(r["n"] / tps(r["n"]) for r in sel) / 3600
        out[L] = dict(n=len(sel), by_key=dict(c), states=len(states), tokens=tok, label_tokens=lab, h_epoch_17b=round(h17, 1), h_epoch_4b=round(h17 / R4B, 1))
        print(f"  ≤{L:>5}: {len(sel):>5} 条（{', '.join(f'{k} {c[k]}' for k in keys)}），覆盖状态 {len(states)}/{len(by_state)}；"
              f"{tok / 1e6:.1f} M token（计损失 {lab / tok:.0%}）；每轮约 1.7B {h17:.1f} h、4B {h17 / R4B:.1f} h")
    if len(a.lens) > 1:
        lo, hi = a.lens[-2], a.lens[-1]
        print(f"\n{hi // 1024}k 比 {lo // 1024}k 多：" + "，".join(f"{k} {out[hi]['by_key'].get(k, 0) - out[lo]['by_key'].get(k, 0):+d}" for k in keys)
              + f"（合计 {out[hi]['n'] - out[lo]['n']:+d} 条，状态 {out[hi]['states'] - out[lo]['states']:+d}）")
    if a.json:
        json.dump(out, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if a.write:
        L, path = int(a.write[0]), pathlib.Path(a.write[1])
        path.parent.mkdir(parents=True, exist_ok=True)
        sel = select(L)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(r["line"] for r in sel)
        print(f"\n→ {path}：{len(sel)} 条（≤ {L} token）")


if __name__ == "__main__":
    main()
