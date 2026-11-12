# Results

One record per decision. Scenario ids: `N*` nominal, `F*` fault, `O*` out-of-distribution, `H*` hard-transient
(`src/scenarios.jsonl`, `src/scenarios_hard.jsonl`); set membership in `src/sets/` and `src/student/sets/`.

## api/ - round-1 experiments (written by `src/evalrun.py`)

| tag (file prefix) | method | scenarios | paper |
|---|---|---|---|
| `main_ds` | workflow, DeepSeek-V4-Pro | test (368 + 182 switching) | Table 4, Fig. 5 |
| `main_base` | NSGA-II + rule isolation / graded slow-down; current rotation table + standard closing (`maker` = optimizer / rules) | test | Table 4 |
| `ood_ds`, `ood_base` | as above | ood150 | Table 4 |
| `main_flash` | workflow, DeepSeek-V4-Flash | test (control60 subset used in Table S7) | Table S7, Fig. 8 |
| `ctrl_kimi`, `ctrl_qwen_max`, `ctrl_qwen_plus`, `ctrl_kimi_k2`, `ctrl_gpt_sol`, `ref_gpt` | workflow, Kimi-K3.1-Pro, Qwen3.8-Max, Qwen3.7-Plus, Kimi-K2.6, GPT-6.1-Sol, GPT-6-Astra | control60 | Table S7, Fig. 8, Fig. S2-S3, S5 |
| `abl_ds_direct`, `abl_ds_no_prompt`, `abl_ds_no_transient`, `abl_ds_no_envelope` | prompt-only (no tools, no feedback); workflow without one component | ablation100 | Fig. S6 |
| `repro_ds` | workflow, DeepSeek-V4-Pro, serial, 10 repeats | repro30 | Section 3.3, Table 5, Fig. 10 |

Shards (`.s<i>of<k>.jsonl`) are parallel lanes of one batch; a later record with the same key replaces an earlier one
(`reproduce/data.py`). `status` is `ok`, `timeout` (with `not_attempted` for Qwen3.8-Max rotation planning, which was not
issued) or `error` (non-model failures, re-run and excluded). Main fields: `feasible_before_fallback`, `final_feasible`,
`fallback_used`, `violations` (steady-state violations of the delivered plan, `{constraint: amount}`),
`transient_violations`, `tau_violations`, `objectives`, `score` (C*), `decision` (the JSON decision), `tokens`, `latency`
(`wall_s`, `model_s`, `retry_wait_s`), `in_window`, `served_model`, `t_utc`.

`final_feasible` is the run-time judgement (time window checked at 1e-6 h). The paper applies a 0.03 h tolerance to the
stored `violations` (`reproduce/data.py`, `apply_twin_tol`; `python reproduce/rescore_counts.py`).

## hard50/ - hard-transient subset (`src/evalrun.py --safety`)

`hard_ds*` workflow configurations (full, without constraint prompt, without transient re-check, without envelope
screening, all three removed) and `hard_base` (standard rules and NSGA-II with rule isolation), 47 scenarios each:
22 switching-stress and 25 burst scenarios with the isolation water hammer included. `safety_final` and `rounds_safety` hold
the full transient check of the executed plan and of every submitted plan (Table S6, Fig. 6, Fig. 7).

## student/ - student-model evaluations (`src/student/eval_student.py`)

One gzip-compressed JSON line per run (the full trajectory, `schema = distill-traj/1`; `_path` is the original
`<tag>/<task>/<job>.json`).

| tag | model, precision, set | paper |
|---|---|---|
| `dpo_a_s00085_q8_cloud` + `dpo_a_s00085_q8_big_cloud` | v3 (deployed), Q8_0, val_ckpt 60 + val_big 190 = val250 | Section 3.4, Fig. 9a-b |
| `v2_s00312_q8_cloud` + `v2_s00312_q8_big_cloud` | v2, Q8_0, val250 | Section 3.4, Fig. 9b |
| `v2_s00312_q8_r2_cloud` | v2, repeat run on val_ckpt (run-to-run spread) | Section 3.4 |
| `v1_s00295_q8_cloud`, `q4b_s00295_q8_cloud`, `q8b_s00295_q8_cloud` | 1.7B / 4B / 8B trained with the v1 recipe, val_ckpt | Section 3.4, Fig. 9c |
| `v2_s00312_q8_cloud`, `v2_s00312_bf16kv16_cloud`, `v2_s00312_q4km_cloud` | v2 in Q8_0 / bf16 / Q4_K_M, val_ckpt | Section 3.4, Fig. 9d |
| `v3_s00085_q8_repro` | v3, Q8_0 + f16 KV, local RTX 5060 8 GB, repro30 x 10 (same states as `repro_ds`) | Table 5, Section 3.5, Fig. 10 |
| `v3_s00085_{q8,bf16,q4km}_kvq8` | v3 in three precisions, local, val_ckpt (q8_0 KV cache) | Section 3.4 |
| `v3_s00085_q8_cpu` | v3 Q8_0 on 8 CPU threads, 9 runs | Section 3.5 |

`*_cloud` runs were made on the rented GPU (llama.cpp built from the same release); latency is reported only from the
local runs. `replaced_contention/` holds the 95 records that were re-run because another program was using the GPU
(selection by timing features only, listed in `rerun_flagged.txt`). `gpu_monitor/` contains the nvidia-smi (0.5 s) and
WDDM (1 s) logs of the local runs. `opt_ref*.json` cache the NSGA-II references used to judge deficit and C*.
