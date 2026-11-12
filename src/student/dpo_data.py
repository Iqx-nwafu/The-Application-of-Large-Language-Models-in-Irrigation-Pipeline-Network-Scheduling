"""DPO 对 → 训练用 token 文件（student 环境，CPU）。prompt+chosen 与 prompt+rejected 各按 sft_data.Encoder 整段渲染（同 SFT：非思考模式、
tools 进模板、train=false 的出错轮不计损失），两条都 ≤ max_len 且都有带标签 token 才保留；两条 token 完全相同的对丢弃。
保存 torch 文件：list[dict(ids_c, mask_c, ids_r, mask_r, meta)]，ids int32、mask bool；另写统计 JSON。
用法：$PY_ST dpo_data.py --model <分词器目录> --max-len 32768 --out <out.pt> [--stats s.json] [--limit n] <pairs.jsonl ...>"""
import argparse
import collections
import json
import pathlib
import sys
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "patches" / "student"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sft_data import Encoder  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("pairs", nargs="+")
ap.add_argument("--model", required=True)
ap.add_argument("--max-len", type=int, default=32768)
ap.add_argument("--out", required=True)
ap.add_argument("--stats", default=None)
ap.add_argument("--limit", type=int, default=None, help="每个来源最多取 n 对（预检 / 冒烟用）")
a = ap.parse_args()
enc = Encoder(a.model)
data, st = [], collections.OrderedDict()
for p in a.pairs:
    p = pathlib.Path(p)
    if not p.exists():
        print(f"!! 缺 {p}，跳过", flush=True)
        continue
    s = collections.Counter()
    tok = 0
    for line in open(p, encoding="utf-8-sig"):
        if not line.strip():
            continue
        d = json.loads(line)
        s["in"] += 1
        try:
            ic, mc = enc(dict(messages=d["prompt"] + d["chosen"], tools=d.get("tools")))
            ir, mr = enc(dict(messages=d["prompt"] + d["rejected"], tools=d.get("tools")))
        except Exception as e:                                  # 模板里 assistant 段数对不上等（Encoder 内有断言）
            s["encode_error"] += 1
            s["_err_" + type(e).__name__] += 1
            continue
        if len(ic) > a.max_len or len(ir) > a.max_len:
            s["too_long"] += 1
            continue
        if not any(mc[1:]) or not any(mr[1:]):
            s["no_label"] += 1
            continue
        if ic == ir:
            s["identical"] += 1
            continue
        pre = next((i for i, (x, y) in enumerate(zip(ic, ir)) if x != y), min(len(ic), len(ir)))
        m = d.get("meta") or {}
        data.append(dict(ids_c=torch.tensor(ic, dtype=torch.int32), mask_c=torch.tensor(mc), ids_r=torch.tensor(ir, dtype=torch.int32),
                         mask_r=torch.tensor(mr), meta=dict(src=p.stem if p.stem != "dpo_pairs" else p.parent.name, chosen=m.get("chosen"),
                                                            rejected=m.get("rejected"), reject=m.get("reject"), state=m.get("state"),
                                                            n_c=len(ic), n_r=len(ir), lab_c=int(sum(mc[1:])), lab_r=int(sum(mr[1:])), common=pre)))
        s["kept"] += 1
        tok += len(ic) + len(ir)
        if a.limit and s["kept"] >= a.limit:
            break
    s["tokens_M"] = round(tok / 1e6, 2)
    st[str(p)] = dict(s)
    print(f"{p}：{dict(s)}", flush=True)
tot = dict(pairs=len(data), tokens_M=round(sum(x["meta"]["n_c"] + x["meta"]["n_r"] for x in data) / 1e6, 2),
           label_tokens_M=round(sum(x["meta"]["lab_c"] + x["meta"]["lab_r"] for x in data) / 1e6, 2),
           by_src=dict(collections.Counter(x["meta"]["src"] for x in data)), max_len=a.max_len, model=a.model)
print("合计：", json.dumps(tot, ensure_ascii=False), flush=True)
out = pathlib.Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(data, out)
if a.stats:
    json.dump(dict(total=tot, by_file=st), open(a.stats, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
if not data:
    sys.exit("没有可用的 DPO 对")
