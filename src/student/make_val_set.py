"""选 checkpoint 用的 val 分层子集（固定种子，生成一次即可，勿重跑覆盖）：只取 split == val（不碰 test / ood）。
  故障：三类各 n_fault/3 个，类型内按故障所在分干管轮转抽（val 无 J7，J7 留给 ood）；
  分组、切换：按灌水次数（需水情景，定额随次数变）轮转抽，切换组对在 0–28 里不放回均匀分配。
隔离后无可调度支管的退化故障（同 O00237）不入选。作业格式与 distill/plans 相同，eval_student.py 直接跑。
  python src/student/make_val_set.py [--n-switch 20 --n-rotation 10 --n-fault 30] [--out src/student/sets/val_ckpt.json]"""
import argparse
import collections
import json
import pathlib
import sys
import numpy as np

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import teacher                                                       # noqa: E402

SEED = 20261103


def round_robin(groups, n, rng):
    """groups: {层: [场景]}；层序与层内顺序都先洗牌，再轮转各取一个，直到 n 个。"""
    keys = list(groups)
    rng.shuffle(keys)
    pools = {k: list(rng.permutation(groups[k])) for k in keys}
    out = []
    while len(out) < n and any(pools.values()):
        for k in keys:
            if pools[k] and len(out) < n:
                out.append(pools[k].pop())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-switch", type=int, default=20)
    ap.add_argument("--n-rotation", type=int, default=10)
    ap.add_argument("--n-fault", type=int, default=30)
    ap.add_argument("--out", default=str(HERE / "sets" / "val_ckpt.json"))
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    S = [json.loads(l) for l in open(ROOT / "scenarios.jsonl", encoding="utf-8")]
    nom = [s for s in S if s["split"] == "val" and s["kind"] == "nominal"]
    fau = [s for s in S if s["split"] == "val" and s["kind"] == "fault"
           and teacher.make_state("fault_response", dict(s, iso_transient=True), rng)[0].valves]
    by_event = collections.defaultdict(list)
    for s in nom:
        by_event[s["irrigation_event"]].append(s["id"])
    jobs = []
    rot = round_robin(by_event, a.n_rotation, rng)
    jobs += [dict(job_id=f"ro_{i}_s0", task="rotation_plan", scenario=i, variant={}, sample=0, seed=SEED) for i in rot]
    sw = round_robin(by_event, a.n_switch, rng)
    pairs = [int(p) for p in rng.permutation(np.resize(np.arange(29), a.n_switch))]
    jobs += [dict(job_id=f"sw_{i}_p{p:02d}_s0", task="switch", scenario=i, variant=dict(pair=p), sample=0, seed=SEED) for i, p in zip(sw, pairs)]
    for t in ("burst", "leak", "stuck"):
        by_sub = collections.defaultdict(list)
        for s in fau:
            if s["faults"][0]["type"] == t:
                by_sub[s["faults"][0].get("submain")].append(s["id"])
        jobs += [dict(job_id=f"fa_{i}_t0_s0", task="fault_response", scenario=i, variant=dict(t=0), sample=0, seed=SEED)
                 for i in round_robin(by_sub, a.n_fault // 3, rng)]
    by_id = {s["id"]: s for s in S}
    summary = dict(
        switch_events=collections.Counter(by_id[j["scenario"]]["irrigation_event"] for j in jobs if j["task"] == "switch"),
        rotation_events=collections.Counter(by_id[j["scenario"]]["irrigation_event"] for j in jobs if j["task"] == "rotation_plan"),
        fault=collections.Counter((by_id[j["scenario"]]["faults"][0]["type"], by_id[j["scenario"]]["faults"][0].get("submain"))
                                  for j in jobs if j["task"] == "fault_response"))
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(seed=SEED, note="val 分层子集，选 checkpoint 用", jobs=jobs), open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(jobs)} 个作业 → {out}")
    for k, v in summary.items():
        print(k, dict(sorted(v.items(), key=str)))


if __name__ == "__main__":
    main()
