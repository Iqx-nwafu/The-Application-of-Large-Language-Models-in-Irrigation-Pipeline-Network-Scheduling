"""方案 C（学生自采样 + 仿真拒绝采样，零 API 费）的采样作业：从候选池里挑训练划分的故障状态，每个状态采 k 次。
默认优先"难状态"：方案 A 筛掉之后一条可用轨迹都不剩的状态（教师只有长循环解法），再补上剩下的故障状态，直到 --n-states。
作业沿用 plan_10000 的故障时刻变体，sample 号从 10 起（与教师样本 0/1 区分），eval_student 按 seed + sample 换随机种子。
  python src\\student\\make_selfsample_set.py <候选池目录> [--n-states 200] [--k 4] [--max-steady 6] [--out src/student/sets/self_c1.json]
跑：eval_student.py --set self_c1 --tag v1_self_c1 --temperature 0.6 …；过滤：to_sft.py --src results/student/v1_self_c1 --repair …（同教师口径）。"""
import argparse
import collections
import json
import pathlib
import random

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent

ap = argparse.ArgumentParser()
ap.add_argument("pool")
ap.add_argument("--n-states", type=int, default=200)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--max-steady", type=int, default=6)
ap.add_argument("--seed", type=int, default=20261107)
ap.add_argument("--out", default=str(HERE / "sets" / "self_c1.json"))
a = ap.parse_args()
pool = pathlib.Path(a.pool)
idx = {(r["job_id"], r["teacher"]): r for r in map(json.loads, open(pool / "index.jsonl", encoding="utf-8"))}
short, every = collections.defaultdict(bool), {}
for line in open(pool / "sft_train.jsonl", encoding="utf-8"):
    r = json.loads(line)
    m = r["meta"]
    if m["task"] != "fault_response":
        continue
    st = tuple(idx[(m["job_id"], m["teacher"])]["state"])
    every[st] = m["fault"]
    steady = sum(c["function"]["name"] == "simulate_steady" for x in r["messages"] if x["role"] == "assistant" for c in x.get("tool_calls") or [])
    short[st] |= steady <= a.max_steady
S = {s["id"]: s for s in (json.loads(l) for l in open(ROOT / "scenarios.jsonl", encoding="utf-8"))}
assert all(S[st[1]]["split"] == "train" for st in every)                       # 只用 train 划分
rng = random.Random(a.seed)
hard = sorted(st for st in every if not short[st])
rest = sorted(st for st in every if short[st])
rng.shuffle(hard); rng.shuffle(rest)
pick = (hard + rest)[:a.n_states]
plan = {(j["scenario"], j["variant"]["t"]): j for j in map(json.loads, open(ROOT / "distill" / "plans" / "plan_10000.jsonl", encoding="utf-8"))
        if j["task"] == "fault_response"}
jobs = []
for st in pick:
    p = plan[(st[1], st[2])]
    for k in range(a.k):
        jobs.append(dict(job_id=f"fa_{st[1]}_t{st[2]}_s{10 + k}", task="fault_response", scenario=st[1], variant=p["variant"],
                         sample=10 + k, seed=p["seed"]))
out = pathlib.Path(a.out)
json.dump(dict(seed=a.seed, note=f"方案 C 自采样：{len(pick)} 个故障状态（难状态 {min(len(hard), len(pick))}）× {a.k} 次", jobs=jobs),
          open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"故障状态 {len(every)}，方案 A 后无可用轨迹的难状态 {len(hard)}；选 {len(pick)} 个 × {a.k} = {len(jobs)} 个作业 → {out}")
print("所选状态的故障类型：", dict(collections.Counter(every[st] for st in pick)))
