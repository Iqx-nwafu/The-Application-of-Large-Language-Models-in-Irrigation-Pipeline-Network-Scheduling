"""v3（部署模型 = DPO a s00085）本机导出，流程与云上逐步一致：
  ① 底座 Qwen3-1.7B + v2 s00312 LoRA 合并（bf16，CPU）→ v2_merged_s00312（DPO 适配器的底座，同云上 dpo_prep.sh）；
  ② DPO 适配器拷到工作目录、把 adapter_config 的底座路径改成本机的合并目录（结果包里的原件不动）；
  ③ 调项目自带的 export_gguf.py：合并 → GGUF bf16 → llama-quantize 出 Q8_0 / Q4_K_M；
  ④ 算三个 GGUF 的 sha256 写 sha256.txt，Q8_0 与云上评测用的那份比前缀（云上只留了前 16 位）。
用法：python src\\deploy\\export_v3.py    （只用 CPU，约 10–20 分钟，占磁盘约 10 GB；可重复运行，已有的步骤跳过）"""
import hashlib
import os
import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = pathlib.Path(os.environ.get("BASE_MODEL", ROOT / "models" / "Qwen3-1.7B"))
RUNS = pathlib.Path(os.environ.get("ADAPTER_DIR", ROOT / "models" / "adapters"))
V2, DPO = RUNS / "v2_sft_s00312", RUNS / "v3_dpo_s00085"          # adapters_v2_v3.tar.gz of the release, unpacked into models/
WORK = pathlib.Path(os.environ.get("EXPORT_WORK", ROOT / "work" / "v3_export"))
MERGED, ADAPTER = WORK / "v2_merged_s00312", WORK / "dpo_a_s00085"
OUT = pathlib.Path(os.environ.get("V3_GGUF_DIR", ROOT / "models" / "gguf" / "v3_dpo_a_s00085"))
EXPORT = ROOT / "src" / "student" / "export_gguf.py"
CLOUD_Q8_PREFIX = "2efe34055d3b5e40"
QUANTS = ["Q8_0", "Q4_K_M"]

for p in (BASE / "config.json", V2 / "adapter_model.safetensors", DPO / "adapter_model.safetensors", EXPORT):
    assert p.exists(), f"缺文件：{p}"
free_gb = shutil.disk_usage(WORK.anchor or ".").free / 2**30
assert free_gb > 15, f"磁盘剩余 {free_gb:.0f} GB，不足 15 GB"
WORK.mkdir(parents=True, exist_ok=True)

if not (MERGED / "config.json").exists():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tmp = WORK / "v2_merged_s00312.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    m = AutoModelForCausalLM.from_pretrained(str(BASE), dtype=torch.bfloat16, device_map="cpu")
    m = PeftModel.from_pretrained(m, str(V2)).merge_and_unload()
    m.save_pretrained(tmp)
    AutoTokenizer.from_pretrained(str(BASE)).save_pretrained(tmp)
    del m
    tmp.rename(MERGED)
print("① v2 合并底座：", MERGED, flush=True)

shutil.copytree(DPO, ADAPTER, dirs_exist_ok=True)
cfg = json.load(open(ADAPTER / "adapter_config.json", encoding="utf-8"))
cfg["base_model_name_or_path"] = str(MERGED)
json.dump(cfg, open(ADAPTER / "adapter_config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("② DPO 适配器副本：", ADAPTER, flush=True)

subprocess.run([sys.executable, str(EXPORT), str(ADAPTER), str(OUT), "--quants", *QUANTS], check=True)

lines, q8 = [], None
for name in ["model-bf16.gguf"] + [f"model-{q}.gguf" for q in QUANTS]:
    h = hashlib.sha256()
    with open(OUT / name, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    lines.append(f"{h.hexdigest()}  {name}  {(OUT / name).stat().st_size}")
    if name == "model-Q8_0.gguf":
        q8 = h.hexdigest()
(OUT / "sha256.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
print("\n".join(lines))
same = q8.startswith(CLOUD_Q8_PREFIX)
print(f"EXPORT_V3_OK q8_same_as_cloud={same}（本机 {q8[:16]}，云上 {CLOUD_Q8_PREFIX}）")
