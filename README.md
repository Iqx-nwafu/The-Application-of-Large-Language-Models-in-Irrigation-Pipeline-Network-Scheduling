# Transient-in-the-loop LLM scheduling of pressurized drip-irrigation networks

Code, benchmark and per-decision results for the paper

> **A transient-in-the-loop large-language-model workflow and a distilled lightweight student model for real-time safe
> rotation and fault scheduling of pressurized drip-irrigation networks** (manuscript submitted to *Computers and
> Electronics in Agriculture*; citation will be added on publication).

The repository contains everything needed to inspect and recompute the results reported in the paper: the 71-node
benchmark network and its 2,700-scenario library, the steady-state and method-of-characteristics (MOC) transient solvers,
the NSGA-II baseline, the tool-calling workflow with its prompts and safety fallback layer, the distillation, training and
evaluation scripts, and the per-decision records behind every table and figure. Training data and the LoRA adapters of the
deployed student model are attached to the GitHub Release (see [Data and model files](#data-and-model-files)).

Code comments and prompts are in Chinese (the workflow prompts the models in Chinese); this README,
`results/README.md` and `reproduce/README.md` describe everything a reader needs in English.

## Repository layout

```
src/                      simulation, benchmark, workflow and evaluation (run scripts from this folder)
  benchmark71.inp         EPANET model of the 71-node, 57.3 ha self-pressurized drip system (Table 1)
  network.py              network parameters and topology -> benchmark71.inp; current rotation table (30 groups x 4 h)
  hydraulics.py           vectorised steady-state solver for the tree network (emitter outflow, automatic throttling)
  moc.py                  vectorised MOC transient solver written for this study, used in all experiments
  steady.py, transient.py, validate_moc.py
                          checks against EPANET/WNTR, the Joukowsky solution and TSNet (outputs in src/validation/)
  evaluate.py, scoring.py objectives, constraints (Eq. 9-11) and the entropy-weight TOPSIS closeness C* (Eq. 12-13)
  transient_tools.py      transient constraint check (pressure limits, vacuum, minimum closing times)
  optimizer.py            NSGA-II rotation grouping (pymoo); also the run_optimizer tool and the fallback optimizer
  envelope.py             single-valve envelope screening (screen_transient tool, round-1 API experiments only)
  tools.py                tool interface and the four-level safety fallback layer (Table S2)
  teacher.py              constraint-augmented prompt, tool-calling loop, forced full re-check and K-round feedback
  evalrun.py, runbatch.py evaluation harness and batch definitions (batches.json, batches_hard.json)
  scenarios.py            scenario library -> scenarios.jsonl (2,700 scenarios, seed 2026)
  make_sets.py            fixed evaluation sets -> sets/*.json (seed 20261028)
  make_hard.py            hard-transient subset -> scenarios_hard.jsonl, sets/hard50.json
  thresholds.json, valve_templates.json, topsis_reference.json, decision_output.schema.json, prices.json
  distill/                teacher-trajectory generation, filtering (rules R1-R8) and conversion to SFT / DPO data
  student/                SFT / DPO training, GGUF export, llama.cpp evaluation; configs/ and evaluation sets in sets/
  deploy/                 local deployment evaluation of the deployed model (repro30 x 10, precision, CPU probe)
reproduce/                read-only re-judging of the stored results with the paper's time-window tolerance
results/
  api/                    round-1 experiments: DeepSeek workflow, eight API models, ablations, NSGA-II and rule baselines
  hard50/                 hard-transient subset (Table S6, Fig. 6)
  student/                student-model evaluations, one .jsonl.gz per evaluation tag
data/                     distillation ledgers (one row per generated trajectory, with filter verdicts and cost)
```

## Environment

Two Python environments were used (versions as in the paper):

* simulation / workflow / evaluation: Python 3.11, numpy 1.26.4, scipy, pandas 2.2, wntr 1.5.0, tsnet 0.3.1, pymoo 0.6.2,
  openai, jsonschema - `pip install -r requirements/sim.txt`
* student training: Python 3.11, torch 2.11 (CUDA 12.8), transformers 5.22.1, peft 0.21.2, accelerate 1.15 -
  `pip install -r requirements/student.txt`; inference with llama.cpp build b12032.

All experiments ran on one laptop (Intel Core Ultra 9 275HX, 32 GB RAM, RTX 5060 Laptop GPU 8 GB, Windows 11); SFT v2, DPO
and the scale ablation were trained on a rented single NVIDIA RTX PRO 6000 (Linux) with the same scripts. Set
`OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=OMP_NUM_THREADS=1` for batch runs (the GA uses a process pool).
API keys are read only from environment variables (`DEEPSEEK_API_KEY`, `MOONSHOT_API_KEY`, `DASHSCOPE_API_KEY`,
`OPENAI_API_KEY`).

## Feasibility counts with the paper's time-window tolerance

The stored records carry the run-time judgement of the time-window constraint (see
[Scoring conventions](#scoring-conventions-used-in-the-paper)). To obtain the feasibility counts reported in the paper:

```bash
pip install -r requirements/sim.txt
python reproduce/rescore_counts.py          # writes reproduce/out/
```

This reads only the released records, re-judges them with the 0.03 h tolerance and prints the counts next to the run-time
counts: Table 4 (test and OOD sets) and the student evaluations of Sections 3.4-3.5 (val250, scale and precision
comparisons, local repro30 x 10). All other reported quantities (C*, latency, cost, token counts, transient pressures) are
fields of the records and do not depend on the tolerance.

## Re-running the pipeline

Simulation, scenarios and the round-1 experiments (run from `src/`; scripts resolve their files relative to `src/` and
write new results to `src/results/`):

```bash
python network.py && python hydraulics.py && python moc.py      # benchmark INP, steady solver vs WNTR, transient self-test
python validate_moc.py                                            # Joukowsky + TSNet comparison (Fig. 3), ~2 min
python scenarios.py && python make_sets.py && python make_hard.py # scenario library and fixed sets (already provided)
python optimizer.py reference                                     # TOPSIS reference set (topsis_reference.json, provided)
python envelope.py build                                          # envelope library for screen_transient (~2.5 min)
python runbatch.py --list                                         # round-1 batches (baselines, API models, ablations)
python runbatch.py base_test ds_test ds_repro                     # e.g. baselines, DeepSeek workflow, serial repro30 x 10
python runbatch.py --config batches_hard.json hard_base hard_ds   # hard-transient subset
```

Distillation (from `src/`; the lean tool set without `screen_transient`, the isolation-water-hammer convention for faults
and at most 20 tool calls per round are the defaults of `distill_gen.py`). Switching and rotation-planning jobs ran on
DeepSeek-V4-Flash first and the jobs whose trajectory failed filter rules R1-R8 (Table S3) were redone on V4-Pro; fault jobs ran on V4-Pro directly
(in a pilot on 20 fault states, 2 of 40 Flash trajectories passed the filter, against 28 of 40 V4-Pro
trajectories on the same states). `data/distill_index_teacher.jsonl.gz` lists all
6,586 generated trajectories with model, filter verdict, tokens and cost.

```bash
python distill/distill_gen.py --plan plan_10000 --model deepseek-chat --raw raw_flash --task switch rotation_plan --budget-cny <cap>
python distill/to_sft.py --src distill/raw_flash --out data_flash --max-tokens 32768
python distill/redo_list.py distill/data_flash/index.jsonl > distill/plans/redo.txt
python distill/distill_gen.py --plan plan_10000 --ids distill/plans/redo.txt --budget-cny <cap>          # V4-Pro, writes distill/raw
python distill/distill_gen.py --plan plan_10000 --task fault_response --budget-cny <cap>                 # V4-Pro
python distill/distill_gen.py --plan plan_stress ...                                                     # switching-stress states, same cascade
python distill/to_sft.py --src distill/raw_flash distill/raw --repair --max-tokens 100000000 --max-per-state 99 --out <work>/pool_b2
```

Student training and evaluation (from the repository root; config paths `models/`, `data/train/`, `runs/` are relative to
it; unpack `training_data.tar` into `data/train/` to skip distillation):

```bash
python src/student/build_sft.py <work>/pool_b2 --max-steady 6 --write 32768 data/train/train_32k_steady6.jsonl
python src/student/train_sft.py src/student/configs/sft_v1_17b.json             # v1 (4,717 teacher trajectories)
python src/student/eval_student.py --set self_c1 --tag v1_self_c1 --temperature 0.6 --abort-repeats 3 --no-ref   # self-sampling
python src/student/train_sft.py src/student/configs/sft_v2_17b.json             # v2 (+ 261 self-generated, from the base)
python src/student/dpo_data.py --model models/Qwen3-1.7B --max-len 32768 --out data/train/dpo_pairs.pt <three pair files>
python src/student/train_dpo.py src/student/configs/dpo_v3.json --max-steps 85  # v3; base and reference = merged v2 weights
python src/student/export_gguf.py <adapter dir> <out dir> --quants Q8_0 Q4_K_M
python src/student/eval_student.py --set val_ckpt --tag <tag> --port 8091       # val_ckpt (60) + val_big (190) = val250
python src/deploy/export_v3.py && python src/deploy/run_local_v3.py chain        # deployed GGUF; repro30 x 10, precision, CPU
```

`models/v2_merged_s00312` (the DPO base) is step 1 of `src/deploy/export_v3.py`. The DPO run used a 144-step cosine
schedule and was stopped at step 85 (1.18 epochs), hence `--max-steps 85`. The deployed checkpoint was selected among
the DPO checkpoints by a rule fixed before evaluation, on the 60 checkpoint-comparison scenarios of val250 (`val_ckpt`):
most qualified fault scenarios, then most fault plans feasible before fallback, then most qualified scenarios overall.
Re-judging with the time-window tolerance does not change the selection.

`student/serve.py` and `deploy/run_local_v3.py` start `llama-server` through Windows WMI (as on the test laptop); on Linux
start `llama-server -m <gguf> --port 8091 -c 32768 -np 1 -ngl 99 -fa on --jinja --seed 0` yourself and run
`eval_student.py` against it.

## Scoring conventions used in the paper

* **Time-window tolerance.** Group durations are written with 0.001 h resolution, so the sum over 40 or more groups can
  exceed the 120 h window by 0.001-0.005 h through rounding alone. The paper judges the time-window constraint with a
  0.03 h tolerance for every method (`evaluate.T_WIN_TOL`). During the experiments the in-loop check used 1e-6 h; the
  released results were scored by applying the 0.03 h tolerance to the stored violation records (rows whose only violation
  is `T_window <= 0.03 h`, with no transient or closing-time violation). `reproduce/` applies exactly this rule; setting
  `T_WIN_TOL = 1e-6` (or `data.CONFIG["twin_tol"] = None`) gives the run-time judgement. The re-judged overruns are all
  0.000-0.005 h; for the round-1 results any tolerance between 0.005 and 0.045 h gives the same counts (the three genuine
  overruns are 0.049, 0.493 and 0.625 h).
* **Isolation-plan consistency (strict).** Fault plans whose declared isolation contradicts the remaining plan count as
  infeasible before fallback; the safety layer (the same code as the NSGA-II + rule-isolation baseline) supplies the final
  plan.
* **Time-outs** count as late (real-time rate) and are excluded from feasibility and C*. Qwen3.8-Max rotation-planning
  requests that were not issued are marked `not_attempted`.
* Scenario **O00237** (all valves of the event isolated, nothing left to schedule) is excluded from all statistics.
* **Latency** is reported only from serial runs (teacher `repro_ds`, student `v3_s00085_q8_repro`); other batches ran
  concurrently with the GA pool.

## Notes on this release

* Code is the code that produced the results, with three changes that do not alter any reported number: the
  time-window tolerance above; the steady-state `PN_PVC` check in `evaluate.py`, which had been swallowed by an inline
  comment (source heads are at most 35 m, far below the 61.2 m rating, so it could never bind); and a portable file lock
  in `optimizer.py` (msvcrt on Windows, fcntl on Linux). Machine-specific paths were replaced by repository-relative
  defaults or environment variables (`LLAMA_SERVER`, `LLAMA_SRC`, `LLAMA_BIN`, `V3_GGUF_DIR`, ...).
* Round-1 API experiments (Table 4, Table S7, hard50) used the original tool set including `screen_transient`, at most 12
  tool calls per round, and the post-isolation rescheduling convention for faults; distillation and all student evaluations
  used the lean tool set, at most 20 calls and the isolation-water-hammer convention (Section 2.3, Table 5 note).
* Greedy decoding in llama.cpp is not bit-reproducible across runs on GPU; the repeat run `v2_s00312_q8_r2_cloud` shows
  the run-to-run spread (about +/-4 of 60 scenarios). API model versions change over time; every round-1 record stores the
  served model name and token counts. Round-1 calls were made between 2026-10-28 and 2026-11-02; Kimi K2 models are
  scheduled to be retired at the end of 2026.

## Data and model files

Attached to the GitHub Release `paper-v1` (checksums in `SHA256SUMS.txt` of the release):

| asset | content |
|---|---|
| `training_data.tar` | SFT sets (4,717 teacher + 261 self-generated trajectories), DPO pair files (810 teacher + 132 self-generated pairs; 570 fit in 32k tokens and were used), and the 5,450-trajectory candidate pool before length / long-loop screening |
| `adapters_v2_v3.tar.gz` | LoRA adapters of SFT v2 (step 312) and of the deployed DPO model v3 (step 85) |

Base model: [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (Apache-2.0). The deployed Q8_0 GGUF is rebuilt with
`src/deploy/export_v3.py` (merge v2 into the base, merge v3 on top, export with llama.cpp b12032); its SHA-256 starts with
`2efe34055d3b5e40`.

**AI-generated content.** The teacher trajectories were generated by DeepSeek-V4-Pro and DeepSeek-V4-Flash through the
DeepSeek API (2026-11-02 to 2026-11-07 UTC), whose terms allow using outputs to train other models; the self-training
trajectories and all student-evaluation records were generated by the authors' fine-tuned Qwen3-1.7B models, and the
round-1 records contain decisions of the eight API models. Every training record carries `meta.ai_content` with the
generating model and UTC time.

## Licenses

* Code: MIT (`LICENSE`).
* Benchmark, scenarios, results and training data: CC BY 4.0 (`DATA_LICENSE.md`); use of the DeepSeek-generated
  trajectories is additionally subject to the DeepSeek terms of service.
* LoRA adapters: Apache-2.0, following the Qwen3 base model.

## 中文简介

本仓库为论文《瞬态在环的大语言模型工作流与蒸馏轻量化学生模型：有压滴灌管网轮灌与故障的实时安全调度》的代码与数据：71 节点基准管网与
2,700 个场景、稳态与特征线法瞬态求解器、NSGA-II 基线、工作流提示与工具接口、蒸馏与训练脚本，以及正文全部表图对应的逐条结果。
`python reproduce/rescore_counts.py` 只读地按正文口径重判并给出可行数。时间窗约束按 0.03 h 数值容差判定（对所有方法一致），实验运行时闭环内用的是
1e-6 h，正文数字由已存越限记录按该容差重判得到，见上文 "Scoring conventions"。训练数据与学生模型适配器在 Release 附件中，蒸馏数据逐条
标注为人工智能生成内容（生成模型与 UTC 时间）。
