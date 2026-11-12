"""蒸馏训练场景抽样 → distill/plans/plan_<N>.jsonl（每行一个生成任务 job）。

场景池：scenarios.jsonl 里 split == "train" 的 1644 个（标称 828、故障 816）。
  - test 368、ood150 本来就不在 train，这里再对 sets/*.json 做一次断言；
  - val 388 不进训练，留给学生模型选 checkpoint / 早停；
  - ood 全部 300 不进训练（只排除 ood150 不够：其余 150 个与它共享留出位置 J7、留出需水、δ = 0.20、组合故障，进训练就泄漏了分布外属性）。
任务配比 switch : rotation_plan : fault_response = 0.4 : 0.2 : 0.4（RATIO），各任务的 job 序列是"轮次 × 分层系统序"，
任一前缀在各层上近似按比例，所以 5000 ⊂ 10000 ⊂ 15000 嵌套：先跑小档、再扩大档时已生成的轨迹全部沿用。
  switch          标称场景 × 基线轮灌表 29 个相邻组对；第 r 轮场景 j 取组对 (j + r) mod 29，(场景, 组对) 不重复，组对均衡
  rotation_plan   标称场景逐轮重采样（教师温度 0.6），第 r 轮 = 第 r 个样本；同一场景 ≥ 2 个样本供 best-of-n 与 DPO 配对
  fault_response  三类故障（爆管 / 泄漏 / 卡死）等份；类型内按 管级或卡死模式 × 分干管位置 分层；
                  状态 = 场景 × 故障时刻变体（t0 为原时刻，t1… 按 scenarios.sample_timing 重采），每个状态 2 个样本

python distill/plan.py                    # 写 plan_5000 / plan_10000 / plan_15000 并打印配比
"""
import collections
import json
import pathlib
import sys
import numpy as np

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import scenarios as scmod                                   # noqa: E402
from make_sets import _sys_keys                             # noqa: E402

SEED = 20261029
RATIO = {"switch": 0.4, "rotation_plan": 0.2, "fault_response": 0.4}
N_PAIRS = 29                                                # 基线轮灌表 30 组 → 29 个相邻组对（teacher.make_state）
FAULT_SAMPLES = 2
TIERS = (5000, 10000, 15000)
N_MAX = max(TIERS)


def load_pool():
    S = [json.loads(l) for l in open(ROOT / "scenarios.jsonl", encoding="utf-8")]
    held = set()
    for fn in (ROOT / "sets").glob("*.json"):
        d = json.load(open(fn, encoding="utf-8"))
        held |= set(d["nominal"]) | set(d["fault"])
    pool = [s for s in S if s["split"] == "train"]
    assert not held & {s["id"] for s in pool}, "训练池与正式实验场景集重叠"
    assert all(not s["ood_tags"] and len(s.get("faults", [])) <= 1 for s in pool)
    return pool


def fault_sub(f):
    return (f.get("pipe_class") or f.get("mode"), "main" if f.get("submain") is None else (f["submain"] - 1) // 3)


def nominal_order(nom, rng):
    return _sys_keys(nom, lambda s: (s["irrigation_event"], s["perturbation"]["delta"]), rng)


def switch_jobs(nom, n, rng):
    order, jobs = nominal_order(nom, rng), []
    for r in range(N_PAIRS):
        for j, s in enumerate(order):
            k = (j + r) % N_PAIRS
            jobs.append(dict(task="switch", scenario=s["id"], variant=dict(pair=k), sample=0, job_id=f"sw_{s['id']}_k{k:02d}"))
            if len(jobs) == n:
                return jobs
    return jobs


def rotation_jobs(nom, n, rng):
    order, jobs, r = nominal_order(nom, rng), [], 0
    while len(jobs) < n:
        jobs += [dict(task="rotation_plan", scenario=s["id"], variant={}, sample=r, job_id=f"ro_{s['id']}_s{r}") for s in order]
        r += 1
    return jobs[:n]


def fault_jobs(flt, n, rng):
    by_type = collections.defaultdict(list)
    for s in flt:
        by_type[s["faults"][0]["type"]].append(s)
    n_states = -(-n // FAULT_SAMPLES)
    per_type = {t: -(-n_states // len(by_type)) for t in by_type}
    streams = {}
    for t, scs in sorted(by_type.items()):
        order = _sys_keys(scs, lambda s: fault_sub(s["faults"][0]), rng)
        st, v = [], 0
        while len(st) < per_type[t]:
            for s in order:
                if v == 0:
                    timing = None                                       # 原时刻
                else:
                    r = np.random.default_rng([SEED, int(s["id"][1:]), v])
                    timing = scmod.sample_timing(r)
                st.append((s, v, timing))
            v += 1
        streams[t] = st[:per_type[t]]
    jobs = []
    for i in range(max(map(len, streams.values()))):                    # 三类交替，前缀等份
        for t in sorted(streams):
            if i < len(streams[t]):
                s, v, timing = streams[t][i]
                for smp in range(FAULT_SAMPLES):
                    jobs.append(dict(task="fault_response", scenario=s["id"], variant=dict(t=v, timing=timing), sample=smp,
                                     job_id=f"fa_{s['id']}_t{v}_s{smp}"))
    return jobs[:n]


def build(pool):
    rng = np.random.default_rng(SEED)
    nom = [s for s in pool if s["kind"] == "nominal"]
    flt = [s for s in pool if s["kind"] == "fault"]
    return {"switch": switch_jobs(nom, round(N_MAX * RATIO["switch"]), rng),
            "rotation_plan": rotation_jobs(nom, round(N_MAX * RATIO["rotation_plan"]), rng),
            "fault_response": fault_jobs(flt, round(N_MAX * RATIO["fault_response"]), rng)}


def main():
    pool = load_pool()
    by_id = {s["id"]: s for s in pool}
    full = build(pool)
    (HERE / "plans").mkdir(exist_ok=True)
    for N in TIERS:
        jobs = []
        for t, js in full.items():
            jobs += js[:round(N * RATIO[t])]
        with open(HERE / "plans" / f"plan_{N}.jsonl", "w", encoding="utf-8") as f:
            for j in jobs:
                f.write(json.dumps(dict(j, seed=SEED), ensure_ascii=False) + "\n")
        c = collections.Counter(j["task"] for j in jobs)
        ft = collections.Counter((by_id[j["scenario"]]["faults"][0]["type"],) + fault_sub(by_id[j["scenario"]]["faults"][0])[:1]
                                 for j in jobs if j["task"] == "fault_response")
        pairs = collections.Counter(j["variant"]["pair"] for j in jobs if j["task"] == "switch")
        print(f"plan_{N}: {dict(c)}；场景数 标称 {len({j['scenario'] for j in jobs if j['task'] != 'fault_response'})} "
              f"故障 {len({j['scenario'] for j in jobs if j['task'] == 'fault_response'})}；"
              f"组对覆盖 {len(pairs)}（每对 {min(pairs.values())}–{max(pairs.values())} 次）")
        print("   故障:", dict(sorted(ft.items())))


if __name__ == "__main__":
    main()
