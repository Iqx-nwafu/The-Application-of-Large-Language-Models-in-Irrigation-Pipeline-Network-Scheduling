"""学生模型正式 LoRA SFT。显存做法沿用 smoke_lora.py（2026-11-02 冒烟：1.7B bf16 24k 峰值 5.70 GB、868 tok/s）：
逐层重算检查点 + 层输入存锁页内存、MLP 与 RMSNorm 分段检查点、peft 层不转 fp32、分块交叉熵（只算有标签的位置）、显存硬上限。
batch = 1 条轨迹，accum 条累一个优化步，损失按组内有标签 token 平均；逐轮屏蔽（train=false 的 assistant 轮 labels=-100）。
超过 max_len 的样本整条丢弃（截断会丢掉最终方案），数量写进 manifest。

用法（student 环境；正式训练用 launch_train.py 后台起）：
  python src\\student\\train_sft.py src\\student\\configs\\sft_v0_17b.json [--out 目录] [--max-steps n]
输出目录：manifest.json（配置、数据、丢弃数、步数）、train_log.jsonl（每优化步）、sample_log.jsonl（每条样本的 loss）、
  ckpt_s<步号>/（按 ckpt_per_epoch 存的 LoRA 适配器，留作在 val 上选 checkpoint）、last/（断点续训：适配器 + 优化器 + 进度，重跑自动续）。
  目录里放一个 STOP 文件：当前优化步做完、存好 last/ 后退出（让出 GPU），原命令重跑即续。
"""
import argparse
import collections
import hashlib
import json
import math
import pathlib
import random
import shutil
import time
import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.checkpoint import checkpoint
from transformers import AttentionInterface, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from sft_data import Encoder

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--out", default=None, help="覆盖配置里的 out_dir（试跑用）")
ap.add_argument("--max-steps", type=int, default=None, help="本次最多跑到第 n 个优化步，存 last/ 后退出（试跑、续训测试）")
a = ap.parse_args()
C = json.load(open(a.config, encoding="utf-8"))
OUT = pathlib.Path(a.out or C["out_dir"])
OUT.mkdir(parents=True, exist_ok=True)
L, MC, CH, ACC = C["max_len"], C.get("mlp_chunk", 4096), C.get("ce_chunk", 256), C["accum"]

free, total = torch.cuda.mem_get_info()                                  # 硬上限：超了报 OOM，不让 WDDM 溢出到共享内存（11-02 三次蓝屏的教训）
CAP = min(free - 0.5 * 2**30, C.get("mem_cap_gb", 99) * 2**30)           # mem_cap_gb：再给同机其它程序留余量（11-04 开训时另有截图任务占显卡）
torch.cuda.set_per_process_memory_fraction(CAP / total)
print(f"显存上限 {CAP / 2**30:.2f} GB（空闲 {free / 2**30:.2f} / 总 {total / 2**30:.2f}）", flush=True)
KERNELS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]   # 本机（SM120、Windows）flash 核不可用，禁止退回 math（L×L 显存）
if C.get("flash", False):                                               # 云上 A100（SM80、Linux）：flash 核可用，放在最前
    KERNELS = [SDPBackend.FLASH_ATTENTION] + KERNELS
OFFLOAD, INNER = C.get("offload", True), C.get("inner_ckpt", True)     # 云上显存够：层输入留在 GPU、不再给 MLP/RMSNorm 套分段检查点

# ---------------- 数据 ----------------
encode = Encoder(C["model"])
data, dropped, sha = [], collections.Counter(), hashlib.sha1()
for p in C["data"]:
    sha.update(open(p, "rb").read())
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        ids, mask = encode(r)
        ids, mask = torch.tensor(ids, dtype=torch.int32), torch.tensor(mask)   # 张量存：5 千条 × 1 万 token 用 list 存要近 4 GB 内存
        if len(ids) > L or not mask[1:].any():
            dropped[r["meta"]["task"]] += 1
            continue
        data.append((ids, mask, dict(job_id=r["meta"]["job_id"], task=r["meta"]["task"], fault=r["meta"].get("fault"))))
N = len(data)
SPE = math.ceil(N / ACC)                                                 # 每轮优化步数（最后一组可不满）
TOTAL = C["epochs"] * SPE
CKPTS = sorted({round(e * SPE + j * SPE / C.get("ckpt_per_epoch", 1)) for e in range(C["epochs"]) for j in range(1, C.get("ckpt_per_epoch", 1) + 1)})
ntok = sum(len(x[0]) for x in data)
print(f"样本 {N}（{dict(collections.Counter(m['task'] for *_, m in data))}），超长或无标签丢弃 {dict(dropped)}；每轮 {ntok / 1e6:.1f} M token、"
      f"{SPE} 步，共 {TOTAL} 步，存 checkpoint 于 {CKPTS}", flush=True)

# ---------------- 模型 ----------------
HC = C.get("attn_head_chunks", 1)


def sdpa_heads(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kw):
    """注意力按 kv 头分 HC 块各调一次 sdpa（只用于整段因果、无 mask 的训练前向）。高效注意力核反向按 3 × T × q 头数分配 fp32 梯度缓冲：
    4B 32k 一次要 1.5 GB（11-03 冒烟 OOM 在这里），分块后每块单独分配、用完即释放。"""
    n, g = k.shape[1] // HC, q.shape[1] // k.shape[1]                    # q 头 i 用 kv 头 i // g（同 repeat_kv）
    out = [F.scaled_dot_product_attention(q[:, i * g:(i + n) * g], k[:, i:i + n], v[:, i:i + n], is_causal=True, scale=scaling, enable_gqa=True)
           for i in range(0, k.shape[1], n)]
    return torch.cat(out, 1).transpose(1, 2).contiguous(), None


AttentionInterface.register("sdpa_heads", sdpa_heads)
torch.manual_seed(C["seed"])
kw = dict(dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa_heads" if HC > 1 else "sdpa")
if C["quant"] == "4bit":
    kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(C["model"], **kw)
model.enable_input_require_grads()                                       # 层输入要带梯度，Offload 的反向才会被调用
lc = C["lora"]
model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc.get("dropout", 0.0),
                                         target_modules=lc.get("target", "all-linear"), task_type="CAUSAL_LM"))
for m in model.modules():
    if hasattr(m, "lora_A"):
        m.cast_input_dtype_enabled = False                                 # peft 默认把层输入转 fp32 存给反向（24k 时注意力段多约 0.8 GB），改由 autocast 做 bf16 矩阵乘
model.train()                                                              # from_pretrained 返回 eval 模式；不开 train，梯度检查点不生效（11-02 蓝屏根因）
params = [p for p in model.parameters() if p.requires_grad]
opt = (bnb.optim.AdamW8bit if C["quant"] == "4bit" else torch.optim.AdamW)(params, lr=C["lr"], weight_decay=C.get("weight_decay", 0.0))
W = max(1, round(C.get("warmup_frac", 0.03) * TOTAL))
MIN = C.get("min_lr_frac", 0.1)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / W if s < W else
                                          MIN + (1 - MIN) * 0.5 * (1 + math.cos(math.pi * (s - W) / max(1, TOTAL - W))))
head = model.get_output_embeddings()
inner = model.base_model.model.model


class Offload(torch.autograd.Function):
    """一层的重算检查点：前向不建图，层输入拷到锁页内存（24k 时 28 层共约 2.8 GB），反向拷回重算。"""
    @staticmethod
    def forward(ctx, h, layer, pe):
        ctx.layer, ctx.pe = layer, pe
        ctx.cpu = torch.empty(h.shape, dtype=h.dtype, pin_memory=True).copy_(h, non_blocking=True) if OFFLOAD else h.detach()
        with torch.no_grad():
            return layer(h, position_embeddings=pe)

    @staticmethod
    def backward(ctx, g):
        h = ctx.cpu.to("cuda", non_blocking=True).requires_grad_()
        ctx.cpu = None                                                     # 图要等下一步 loss 重新赋值才释放，不清掉锁页内存会翻倍
        with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            torch.autograd.backward(ctx.layer(h, position_embeddings=ctx.pe), g)
        return h.grad, None, None


def mlp(self, x):
    f = lambda t: self.down_proj(self.act_fn(self.gate_proj(t)) * self.up_proj(t))
    parts = [x[:, i:i + MC] for i in range(0, x.shape[1], MC)]
    return torch.cat([checkpoint(f, t, use_reentrant=False) if torch.is_grad_enabled() else f(t) for t in parts], 1)


def ckpt(fwd):
    """RMSNorm 按 MC 分段套检查点：fp32 中间量不留到反向，重算时也只有一段（4B 的 q_norm 整段 fp32 为 384 MB）。"""
    return lambda self, x: torch.cat([checkpoint(fwd, self, t, use_reentrant=False) if torch.is_grad_enabled() else fwd(self, t)
                                      for t in x.split(MC, 1)], 1)


if INNER:
    type(inner.layers[0].mlp).forward = mlp
    type(inner.norm).forward = ckpt(type(inner.norm).forward)


def hidden(x):
    h = inner.embed_tokens(x)
    pe = inner.rotary_emb(h, torch.arange(h.shape[1], device="cuda")[None])
    for layer in inner.layers:
        h = Offload.apply(h, layer, pe)
    return inner.norm(h)[0]


def chunk_loss(h, y):
    return F.cross_entropy(head(h).float(), y, reduction="sum")


def sample_loss(ids, mask, n_group):
    """一条轨迹前向 + 反向；返回该条有标签 token 的平均 loss。词表投影只算有标签的位置（约占 token 的三成）。"""
    x = ids.to("cuda", torch.long)[None]
    pos = mask[1:].nonzero().squeeze(1).to("cuda")                       # 位置 t 预测 t+1
    y = x[0, 1:].index_select(0, pos)
    with sdpa_kernel(KERNELS), torch.autocast("cuda", dtype=torch.bfloat16):
        h = hidden(x).index_select(0, pos)
        tot = sum(checkpoint(chunk_loss, h[i:i + CH], y[i:i + CH], use_reentrant=False) for i in range(0, len(y), CH))
        (tot / n_group).backward()
    return tot.item() / len(y)


# ---------------- 续训 ----------------
def save_last(step):
    tmp = OUT / "last.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    torch.save(dict(step=step, opt=opt.state_dict(), sched=sched.state_dict(), config=C, cuda_rng=torch.cuda.get_rng_state(),
                    cpu_rng=torch.get_rng_state()), tmp / "state.pt")
    shutil.rmtree(OUT / "last.old", ignore_errors=True)
    if (OUT / "last").exists():
        (OUT / "last").rename(OUT / "last.old")
    tmp.rename(OUT / "last")
    shutil.rmtree(OUT / "last.old", ignore_errors=True)


step = 0
if (OUT / "last" / "state.pt").exists():
    st = torch.load(OUT / "last" / "state.pt", weights_only=False)
    IGN = ("mem_cap_gb", "_note")                                     # 只影响显存上限 / 说明的键不参与比对（11-09 云上放宽 mem_cap_gb 后要能续训）
    same = {k: v for k, v in st["config"].items() if k not in IGN} == {k: v for k, v in C.items() if k not in IGN}
    assert same, "last/ 是用另一份配置训的（除 mem_cap_gb/_note 外有键不同），换输出目录或删掉 last/"
    set_peft_model_state_dict(model, load_file(OUT / "last" / "adapter_model.safetensors"))
    opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
    torch.cuda.set_rng_state(st["cuda_rng"]); torch.set_rng_state(st["cpu_rng"])
    step = st["step"]
    print(f"从 last/ 续训：已完成 {step}/{TOTAL} 步", flush=True)
json.dump(dict(config=C, data_sha1=sha.hexdigest(), n=N, by_task=collections.Counter(m["task"] for *_, m in data), dropped=dropped,
               tokens_per_epoch=ntok, steps_per_epoch=SPE, total_steps=TOTAL, ckpt_steps=CKPTS, cap_gb=round(CAP / 2**30, 2),
               started=time.strftime("%Y-%m-%d %H:%M:%S"), resumed_from=step), open(OUT / f"manifest_{step:05d}.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

# ---------------- 训练 ----------------
flog, slog = open(OUT / "train_log.jsonl", "a", encoding="utf-8"), open(OUT / "sample_log.jsonl", "a", encoding="utf-8")
torch.cuda.reset_peak_memory_stats()
t_run, tok_run = time.time(), 0
end = min(TOTAL, a.max_steps or TOTAL)
while step < end:
    ep, j = divmod(step, SPE)
    order = list(range(N))
    random.Random(C["seed"] + ep).shuffle(order)                         # 每轮重洗；续训时同一轮顺序不变
    group = order[j * ACC:(j + 1) * ACC]
    n_group = sum(int(data[i][1][1:].sum()) for i in group)
    t0, losses = time.time(), []
    for i in group:
        ids, mask, meta = data[i]
        li = sample_loss(ids, mask, n_group)
        losses.append((li, int(mask[1:].sum())))
        slog.write(json.dumps(dict(step=step, epoch=ep, **meta, n_tokens=len(ids), loss=round(li, 4)), ensure_ascii=False) + "\n")
    gn = torch.nn.utils.clip_grad_norm_(params, C.get("clip", 1.0)).item()
    lr = sched.get_last_lr()[0]
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True); torch.cuda.synchronize()
    step += 1
    nt = sum(len(data[i][0]) for i in group)
    tok_run += nt
    rate = tok_run / (time.time() - t_run)
    left = (TOTAL - step) * ntok / SPE / rate
    flog.write(json.dumps(dict(step=step, epoch=round(step / SPE, 3), lr=lr, loss=round(sum(l * n for l, n in losses) / n_group, 4),
                               grad_norm=round(gn, 3), n_tokens=nt, n_label=n_group, sec=round(time.time() - t0, 1), tok_s=round(rate),
                               peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2), t=time.strftime("%m-%d %H:%M:%S"))) + "\n")
    flog.flush(); slog.flush()
    print(f"step {step}/{TOTAL} ep {step / SPE:.2f} loss {sum(l * n for l, n in losses) / n_group:.4f} lr {lr:.2e} gn {gn:.2f} "
          f"{rate:.0f} tok/s peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB 余 {left / 3600:.1f} h {time.strftime('%H:%M:%S')}", flush=True)
    if step in CKPTS:
        model.save_pretrained(OUT / f"ckpt_s{step:05d}")
        print(f"  → ckpt_s{step:05d}（第 {step / SPE:.2f} 轮）", flush=True)
    stop = (OUT / "STOP").exists()
    if step % C.get("save_every", 10) == 0 or step == end or stop:
        save_last(step)
    if stop:
        (OUT / "STOP").unlink()
        print(f"收到 STOP，已存 last/（第 {step} 步）后退出", flush=True)
        break
print(f"{'完成' if step == TOTAL else '暂停'}：{step}/{TOTAL} 步，峰值 {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
