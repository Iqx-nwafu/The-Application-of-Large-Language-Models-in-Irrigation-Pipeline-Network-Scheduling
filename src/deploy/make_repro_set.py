"""学生正式时延 / 可重复性场景集，与教师 repro_ds 用同一批状态（irrig 环境）：
  python src\\deploy\\make_repro_set.py
教师那次（evalrun --set repro30 --repeats 10 --seed 0）每个 (场景, 第 r 次) 的状态 = teacher.make_state(task, sc, default_rng([0, r, 场景号]))：
只有切换用随机数（相邻两组的起始组号 k = rng.integers(29)），分组与故障的状态不随 r 变。这里按同一规则算每次重复的 k，写进学生评测集的
variant.pair（distill_gen.FixedPair 把它原样送回 make_state），并与 results/repro_ds.jsonl 里教师实际切换的组号逐条核对。
写 student/sets/repro30_v3.json（15 标称 ×（切换 + 分组）+ 15 故障，各 10 次 = 450 条）与 cpu_probe_v3.json（val_ckpt 的前 5 切换 / 2 分组 / 2 故障）。"""
import collections
import json
import os
import pathlib
import sys

for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(k, "1")
STAGEA = pathlib.Path(os.environ.get("LOCAL_V3_STAGEA", pathlib.Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(STAGEA)]
os.chdir(STAGEA)
import evalrun                                                        # noqa: E402
import network as net                                                 # noqa: E402

SEED, REPEATS, PLAN_SEED = 0, 10, 20261111
S = {s["id"]: s for s in (json.loads(l) for l in open("scenarios.jsonl", encoding="utf-8"))}
R30 = json.load(open("sets/repro30.json", encoding="utf-8"))
n_groups = len(net.rotation_groups())
assert n_groups == 30, n_groups

jobs = []
for r in range(REPEATS):
    for i in R30["nominal"]:
        k = S[i]["switch_pair"][0] if S[i].get("switch_pair") else int(evalrun.scenario_rng(SEED, S[i], r).integers(n_groups - 1))
        jobs.append(dict(job_id=f"sw_{i}_p{k:02d}_s{r}", task="switch", scenario=i, variant=dict(pair=k), sample=r, seed=PLAN_SEED))
        jobs.append(dict(job_id=f"ro_{i}_s{r}", task="rotation_plan", scenario=i, variant={}, sample=r, seed=PLAN_SEED))
    for i in R30["fault"]:
        jobs.append(dict(job_id=f"fa_{i}_t0_s{r}", task="fault_response", scenario=i, variant=dict(t=0), sample=r, seed=PLAN_SEED))
assert len({j["job_id"] for j in jobs}) == len(jobs) == 450
assert all(S[j["scenario"]]["split"] == "test" for j in jobs)

# 与教师实际切换的组号核对
want = {(j["scenario"], j["sample"]): j["variant"]["pair"] + 1 for j in jobs if j["task"] == "switch"}
same = diff = 0
for line in open("results/repro_ds.jsonl", encoding="utf-8"):
    x = json.loads(line)
    fg = ((x.get("decision") or {}).get("switch") or {}).get("from_group")
    if x["task"] == "switch" and fg is not None:
        ok = want[(x["scenario"], x["repeat"])] == fg
        same, diff = same + ok, diff + (not ok)
print(f"切换组号与教师 repro_ds 核对：一致 {same}，不一致 {diff}")
assert same >= 135 and diff <= 15, "切换状态与教师那次对不上，不写文件"   # 规则不对时一致数只会在 1/29 左右；个别不一致是教师自己把组号写错

sets = pathlib.Path("student/sets")
json.dump(dict(seed=str(PLAN_SEED), note="学生正式时延与可重复性：repro30 的 45 个状态 × 10 次，状态与教师 repro_ds（evalrun --seed 0）逐条相同；test 划分",
               jobs=jobs), open(sets / "repro30_v3.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
V = json.load(open(sets / "val_ckpt.json", encoding="utf-8"))["jobs"]
probe = [j for t, n in (("switch", 5), ("rotation_plan", 2), ("fault_response", 2)) for j in [x for x in V if x["task"] == t][:n]]
json.dump(dict(seed="20261103", note="纯 CPU 探针：val_ckpt 的前 5 切换 / 2 分组 / 2 故障", jobs=probe),
          open(sets / "cpu_probe_v3.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("repro30_v3：", dict(collections.Counter(j["task"] for j in jobs)), "；cpu_probe_v3：", dict(collections.Counter(j["task"] for j in probe)))
print("MAKE_SETS_OK")
