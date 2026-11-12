"""学生模型轨迹级 LoRA DPO（以 SFT v2 合并后的权重为底座与参考模型；云上，2026-11-09 作者 决策链：v2 没拉起就接 DPO）。
显存做法沿用 train_sft.py：逐层重算检查点（Offload）、分块交叉熵（只算有标签位置）、显存硬上限、peft 层不转 fp32。
一对 = prompt+chosen 与 prompt+rejected 两条完整轨迹（dpo_data.py 已编码），序列对数似然 = 有标签（assistant 轮、train=false 的出错轮除外）token 的
logp 之和（同 ETO 的轨迹级 DPO）；参考模型 = 同一模型关掉 LoRA（底座即 v2）。
  loss = mean_pairs[ -log σ(β((logπ_c − logπref_c) − (logπ_r − logπref_r))) ] + nll_coef · Σ(−logπ_c) / Σ n_label_c   （nll_coef=0 为纯 DPO）
accum 对累一个优化步；两条轨迹的图同时保留（fast 写法层输入留 GPU，32k 两条约 7.5 GB），一次 backward。
断点续训 / STOP 文件 / --max-steps 同 train_sft.py。config.deadline（epoch 秒）：到点时存 checkpoint、把 manifest 的 ckpt_steps 改成已存的 checkpoint、退出 0。
用法：$PY_ST train_dpo.py <config.json> [--out 目录] [--max-steps n]"""
import argparse
import collections
import contextlib
import hashlib
import json
import math
import pathlib
import random
import shutil
import time
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--out", default=None)
ap.add_argument("--max-steps", type=int, default=None)
a = ap.parse_args()
C = json.load(open(a.config, encoding="utf-8"))
OUT = pathlib.Path(a.out or C["out_dir"])
OUT.mkdir(parents=True, exist_ok=True)
L, MC, CH, ACC = C["max_len"], C.get("mlp_chunk", 4096), C.get("ce_chunk", 256), C["accum"]
BETA, NLL = C.get("beta", 0.1), C.get("nll_coef", 0.0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cuda":
    free, total = torch.cuda.mem_get_info()
    CAP = min(free - 0.5 * 2**30, C.get("mem_cap_gb", 99) * 2**30)
    torch.cuda.set_per_process_memory_fraction(CAP / total)
    print(f"显存上限 {CAP / 2**30:.2f} GB（空闲 {free / 2**30:.2f} / 总 {total / 2**30:.2f}）", flush=True)
    from torch.nn.attention import sdpa_kernel, SDPBackend
    KERNELS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]
    if C.get("flash", True):
        KERNELS = [SDPBackend.FLASH_ATTENTION] + KERNELS
    kernels = lambda: sdpa_kernel(KERNELS)
else:
    CAP = 0
    kernels = contextlib.nullcontext
autocast = lambda: torch.autocast(DEV, dtype=torch.bfloat16)
OFFLOAD, INNER = C.get("offload", False), C.get("inner_ckpt", False)

# ---------------- 数据 ----------------
data, sha = [], hashlib.sha1()
for p in C["data"]:
    sha.update(open(p, "rb").read())
    for d in torch.load(p):
        if len(d["ids_c"]) <= L and len(d["ids_r"]) <= L:
            data.append(d)
N = len(data)
SPE = math.ceil(N / ACC)
TOTAL = C["epochs"] * SPE
CKPTS = sorted({round(e * SPE + j * SPE / C.get("ckpt_per_epoch", 1)) for e in range(C["epochs"]) for j in range(1, C.get("ckpt_per_epoch", 1) + 1)})
ntok = sum(len(d["ids_c"]) + len(d["ids_r"]) for d in data)
print(f"DPO 对 {N}（{dict(collections.Counter(d['meta']['src'] for d in data))}）；每轮 {ntok / 1e6:.1f} M token、{SPE} 步，共 {TOTAL} 步，"
      f"存 checkpoint 于 {CKPTS}；β={BETA} nll_coef={NLL}", flush=True)

# ---------------- 模型 ----------------
torch.manual_seed(C["seed"])
model = AutoModelForCausalLM.from_pretrained(C["model"], dtype=torch.bfloat16, device_map=DEV, attn_implementation="sdpa")
model.enable_input_require_grads()
lc = C["lora"]
model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc.get("dropout", 0.0),
                                         target_modules=lc.get("target", "all-linear"), task_type="CAUSAL_LM"))
for m in model.modules():
    if hasattr(m, "lora_A"):
        m.cast_input_dtype_enabled = False
model.train()
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=C["lr"], weight_decay=C.get("weight_decay", 0.0))
W = max(1, round(C.get("warmup_frac", 0.1) * TOTAL))
MIN = C.get("min_lr_frac", 0.1)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / W if s < W else
                                          MIN + (1 - MIN) * 0.5 * (1 + math.cos(math.pi * (s - W) / max(1, TOTAL - W))))
head = model.get_output_embeddings()
inner = model.base_model.model.model


class Offload(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, layer, pe):
        ctx.layer, ctx.pe = layer, pe
        ctx.cpu = torch.empty(h.shape, dtype=h.dtype, pin_memory=True).copy_(h, non_blocking=True) if OFFLOAD else h.detach()
        with torch.no_grad():
            return layer(h, position_embeddings=pe)

    @staticmethod
    def backward(ctx, g):
        h = ctx.cpu.to(DEV, non_blocking=True).requires_grad_()
        ctx.cpu = None
        with torch.enable_grad(), autocast():
            torch.autograd.backward(ctx.layer(h, position_embeddings=ctx.pe), g)
        return h.grad, None, None


def mlp(self, x):
    f = lambda t: self.down_proj(self.act_fn(self.gate_proj(t)) * self.up_proj(t))
    parts = [x[:, i:i + MC] for i in range(0, x.shape[1], MC)]
    return torch.cat([checkpoint(f, t, use_reentrant=False) if torch.is_grad_enabled() else f(t) for t in parts], 1)


def ckpt(fwd):
    return lambda self, x: torch.cat([checkpoint(fwd, self, t, use_reentrant=False) if torch.is_grad_enabled() else fwd(self, t)
                                      for t in x.split(MC, 1)], 1)


if INNER:
    type(inner.layers[0].mlp).forward = mlp
    type(inner.norm).forward = ckpt(type(inner.norm).forward)


def hidden(x):
    h = inner.embed_tokens(x)
    pe = inner.rotary_emb(h, torch.arange(h.shape[1], device=DEV)[None])
    for layer in inner.layers:
        h = Offload.apply(h, layer, pe) if torch.is_grad_enabled() else layer(h, position_embeddings=pe)
    return inner.norm(h)[0]


def chunk_nll(h, y):
    return F.cross_entropy(head(h).float(), y, reduction="sum")


def seq_nll(ids, mask):
    """一条轨迹有标签位置的 −Σ logp（建图时每个词表投影块套检查点）；返回 (张量, 标签数)。"""
    x = ids.to(DEV, torch.long)[None]
    pos = mask[1:].nonzero().squeeze(1).to(DEV)
    y = x[0, 1:].index_select(0, pos)
    with kernels(), autocast():
        h = hidden(x).index_select(0, pos)
        if torch.is_grad_enabled():
            tot = sum(checkpoint(chunk_nll, h[i:i + CH], y[i:i + CH], use_reentrant=False) for i in range(0, len(y), CH))
        else:
            tot = sum(chunk_nll(h[i:i + CH], y[i:i + CH]) for i in range(0, len(y), CH))
    return tot, len(y)


def pair_loss(d, n_group, lab_group):
    """一对：参考（关 LoRA、无图）→ 策略两条建图 → DPO 损失 + NLL 正则 → backward。返回日志量。"""
    with torch.no_grad(), model.disable_adapter():
        ref_c, _ = seq_nll(d["ids_c"], d["mask_c"])
        ref_r, _ = seq_nll(d["ids_r"], d["mask_r"])
    nll_c, n_c = seq_nll(d["ids_c"], d["mask_c"])
    nll_r, n_r = seq_nll(d["ids_r"], d["mask_r"])
    z = BETA * ((ref_c - nll_c) - (ref_r - nll_r))                      # logπ − logπref 之差（nll = −logp）
    loss = F.softplus(-z) / n_group + (NLL * nll_c / lab_group if NLL else 0)
    loss.backward()
    return dict(z=round(z.item(), 4), dpo=round(F.softplus(-z).item(), 4), nll_c=round(nll_c.item() / n_c, 4), nll_r=round(nll_r.item() / n_r, 4),
                ref_c=round(ref_c.item() / n_c, 4), ref_r=round(ref_r.item() / n_r, 4))     # nll_* / ref_*：每个有标签 token 的均值（标签数在 meta 的 lab_c / lab_r）


# ---------------- 续训 ----------------
def save_last(step):
    tmp = OUT / "last.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    torch.save(dict(step=step, opt=opt.state_dict(), sched=sched.state_dict(), config=C, rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state() if DEV == "cuda" else None), tmp / "state.pt")
    shutil.rmtree(OUT / "last.old", ignore_errors=True)
    if (OUT / "last").exists():
        (OUT / "last").rename(OUT / "last.old")
    tmp.rename(OUT / "last")
    shutil.rmtree(OUT / "last.old", ignore_errors=True)


def rewrite_manifests(ckpt_steps, **extra):
    for p in OUT.glob("manifest_*.json"):
        m = json.load(open(p, encoding="utf-8"))
        m.update(ckpt_steps=ckpt_steps, **extra)
        json.dump(m, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


step = 0
if (OUT / "last" / "state.pt").exists():
    st = torch.load(OUT / "last" / "state.pt", weights_only=False)
    IGN = ("mem_cap_gb", "_note", "deadline")
    same = {k: v for k, v in st["config"].items() if k not in IGN} == {k: v for k, v in C.items() if k not in IGN}
    assert same, "last/ 是用另一份配置训的（除 mem_cap_gb/_note/deadline 外有键不同），换输出目录或删掉 last/"
    set_peft_model_state_dict(model, load_file(OUT / "last" / "adapter_model.safetensors"))
    opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
    torch.set_rng_state(st["rng"])
    if DEV == "cuda" and st.get("cuda_rng") is not None:
        torch.cuda.set_rng_state(st["cuda_rng"])
    step = st["step"]
    print(f"从 last/ 续训：已完成 {step}/{TOTAL} 步", flush=True)
json.dump(dict(config=C, data_sha1=sha.hexdigest(), n=N, by_src=collections.Counter(d["meta"]["src"] for d in data), tokens_per_epoch=ntok,
               steps_per_epoch=SPE, total_steps=TOTAL, ckpt_steps=CKPTS, cap_gb=round(CAP / 2**30, 2), started=time.strftime("%Y-%m-%d %H:%M:%S"),
               resumed_from=step), open(OUT / f"manifest_{step:05d}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---------------- 训练 ----------------
flog, slog = open(OUT / "train_log.jsonl", "a", encoding="utf-8"), open(OUT / "sample_log.jsonl", "a", encoding="utf-8")
if DEV == "cuda":
    torch.cuda.reset_peak_memory_stats()
peak = lambda: torch.cuda.max_memory_allocated() / 2**30 if DEV == "cuda" else 0.0
t_run, tok_run = time.time(), 0
end = min(TOTAL, a.max_steps or TOTAL)
DEADLINE = C.get("deadline")
saved = [s for s in CKPTS if (OUT / f"ckpt_s{s:05d}" / "adapter_model.safetensors").exists()]
while step < end:
    ep, j = divmod(step, SPE)
    order = list(range(N))
    random.Random(C["seed"] + ep).shuffle(order)
    group = order[j * ACC:(j + 1) * ACC]
    n_group = len(group)
    lab_group = sum(int(data[i]["mask_c"][1:].sum()) for i in group)
    t0, rows = time.time(), []
    for i in group:
        r = pair_loss(data[i], n_group, lab_group)
        rows.append(dict(r, n_c=data[i]["meta"]["lab_c"], n_r=data[i]["meta"]["lab_r"]))
        slog.write(json.dumps(dict(step=step, epoch=ep, **data[i]["meta"], **r), ensure_ascii=False) + "\n")
    gn = torch.nn.utils.clip_grad_norm_(params, C.get("clip", 1.0)).item()
    lr = sched.get_last_lr()[0]
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if DEV == "cuda":
        torch.cuda.synchronize()
    step += 1
    nt = sum(len(data[i]["ids_c"]) + len(data[i]["ids_r"]) for i in group)
    tok_run += nt
    rate = tok_run / (time.time() - t_run)
    left = (TOTAL - step) * ntok / SPE / rate
    acc = sum(r["z"] > 0 for r in rows) / len(rows)
    rec = dict(step=step, epoch=round(step / SPE, 3), lr=lr, dpo=round(sum(r["dpo"] for r in rows) / len(rows), 4),
               margin=round(sum(r["z"] for r in rows) / len(rows) / BETA, 3), acc=round(acc, 3),
               nll_c=round(sum(r["nll_c"] * r["n_c"] for r in rows) / sum(r["n_c"] for r in rows), 4),
               nll_r=round(sum(r["nll_r"] * r["n_r"] for r in rows) / sum(r["n_r"] for r in rows), 4),
               grad_norm=round(gn, 3), n_tokens=nt, sec=round(time.time() - t0, 1), tok_s=round(rate), peak_gb=round(peak(), 2), t=time.strftime("%m-%d %H:%M:%S"))
    flog.write(json.dumps(rec) + "\n"); flog.flush(); slog.flush()
    print(f"step {step}/{TOTAL} ep {step / SPE:.2f} dpo {rec['dpo']:.4f} margin {rec['margin']:.2f} acc {acc:.2f} nll_c {rec['nll_c']:.3f} "
          f"nll_r {rec['nll_r']:.3f} lr {lr:.2e} gn {gn:.2f} {rate:.0f} tok/s peak {rec['peak_gb']:.2f} GB 余 {left / 3600:.1f} h {time.strftime('%H:%M:%S')}", flush=True)
    if step in CKPTS:
        model.save_pretrained(OUT / f"ckpt_s{step:05d}")
        saved.append(step)
        print(f"  → ckpt_s{step:05d}（第 {step / SPE:.2f} 轮）", flush=True)
    stop = (OUT / "STOP").exists()
    late = DEADLINE is not None and time.time() > DEADLINE and step < TOTAL
    if late and step not in saved:                                        # 到时：把当前步存成 checkpoint，manifest 的 ckpt_steps 改成实际存下的
        model.save_pretrained(OUT / f"ckpt_s{step:05d}")
        saved.append(step)
    if late:
        ck = sorted(set(saved))
        rewrite_manifests((ck + ck)[:2] if len(ck) < 2 else ck[-2:], early_stop=dict(step=step, total=TOTAL, t=time.strftime("%Y-%m-%d %H:%M:%S")))
        print(f"到 deadline，提前停止：{step}/{TOTAL} 步，checkpoint {ck}（manifest 已改）", flush=True)
    if step % C.get("save_every", 5) == 0 or step == end or stop or late:
        save_last(step)
    if stop:
        (OUT / "STOP").unlink()
        print(f"收到 STOP，已存 last/（第 {step} 步）后退出", flush=True)
        break
    if late:
        break
print(f"{'完成' if step == TOTAL else '暂停'}：{step}/{TOTAL} 步，峰值 {peak():.2f} GB", flush=True)
