"""LoRA checkpoint → 合并进 bf16 底座 → GGUF（bf16）→ llama-quantize 出 Q8_0 / Q4_K_M 等（同一后端的三档 PTQ，HANDOFF §9 第 8 条）。
  python src\\student\\export_gguf.py <适配器目录（ckpt_s*）> <输出目录> [--quants Q8_0 Q4_K_M] [--keep-hf]
需要与二进制同版本的 llama.cpp 源码脚本：llama.cpp/src-b12032/convert_hf_to_gguf.py 与 gguf-py（二进制包里没有）。
合并在 CPU 上做（1.7B bf16 约 3.4 GB 内存，4B 约 8 GB），不占 GPU。4B 是 QLoRA：训练时底座为 nf4，这里合并进 bf16 底座（论文注明）。
输出 model-bf16.gguf 与 model-<档>.gguf；合并后的 HF 目录默认删掉（每个 checkpoint 3.4 GB），--keep-hf 保留。"""
import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SRC = pathlib.Path(os.environ.get("LLAMA_SRC", "llama.cpp/src-b12032"))      # 云上：环境变量指向同版本源码脚本与编译出的 bin
BIN = pathlib.Path(os.environ.get("LLAMA_BIN", "llama.cpp/b12032/bin"))
STUB = pathlib.Path(os.environ.get("LLAMA_STUB", str(SRC.parent / "stub")))
EXE = ".exe" if os.name == "nt" else ""

ap = argparse.ArgumentParser()
ap.add_argument("adapter")
ap.add_argument("out")
ap.add_argument("--quants", nargs="*", default=["Q8_0"])
ap.add_argument("--keep-hf", action="store_true")
a = ap.parse_args()
adapter, out = pathlib.Path(a.adapter), pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)
base = json.load(open(adapter / "adapter_config.json", encoding="utf-8"))["base_model_name_or_path"]
if not pathlib.Path(base).exists() and os.environ.get("MODELS_DIR"):             # 适配器里记的底座路径不存在时，到 MODELS_DIR 下按目录名找
    base = str(pathlib.Path(os.environ["MODELS_DIR"]) / pathlib.PurePath(base.replace("\\", "/")).name)
gg = out / "model-bf16.gguf"
if not gg.exists():
    hf = out / "hf"
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(hf)
    AutoTokenizer.from_pretrained(base).save_pretrained(hf)
    del model
    subprocess.run([sys.executable, str(SRC / "convert_hf_to_gguf.py"), str(hf), "--outtype", "bf16", "--outfile", str(gg)], check=True,
                   env=dict(os.environ, PYTHONPATH=os.pathsep.join([str(SRC / "gguf-py"), str(STUB)])))       # stub/sentencepiece.py：Qwen3 用不到，免装
    if not a.keep_hf:
        shutil.rmtree(hf)
for q in a.quants:
    if not (out / f"model-{q}.gguf").exists():
        nt = [os.environ["QUANT_THREADS"]] if os.environ.get("QUANT_THREADS") else []     # 云上容器按配额给线程数（默认会按整机核数开）
        subprocess.run([str(BIN / f"llama-quantize{EXE}"), str(gg), str(out / f"model-{q}.gguf"), q, *nt], check=True)
json.dump(dict(adapter=str(adapter.resolve()), base=base, quants=a.quants), open(out / "export.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("→", out, sorted(p.name for p in out.glob("*.gguf")))
