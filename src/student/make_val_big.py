"""扩大验证集（2026-11-10，v2 与 DPO 配对重评用）：只取 split == val 且没进 val_ckpt 的场景（不碰 train / test / ood），分层方式同 make_val_set.py。
  故障：三类各 n_fault/3 个，类型内按故障所在分干管轮转抽；分组、切换：按灌水次数轮转抽，切换组对在 0–28 里均匀分配。
  退化故障（按作业实际的状态构造方式——iso_transient、作业随机数——隔离后无可调度支管）不入选。
选场景用固定种子 20261110；作业字段与 val_ckpt 相同（plan seed 20261103、sample 0），eval_student.py 直接跑。已有文件不覆盖（--force 才重写）。
  $PY_IR make_val_big.py [--n-switch 40 --n-rotation 30 --n-fault 120] [--out $ST/student/sets/val_big.json]"""
import argparse
import collections
import json
import os
import pathlib
import sys

import numpy as np

ST = pathlib.Path(os.environ["ST"])
sys.path[:0] = [str(ST), str(ST / "student")]
import teacher                                                       # noqa: E402
from make_val_set import round_robin                                 # noqa: E402

SEL_SEED, JOB_SEED = 20261110, 20261103


def summary(jobs, by_id):
    sc = lambda j: by_id[j["scenario"]]
    return dict(n=dict(collections.Counter(j["task"] for j in jobs)),
                fault=dict(sorted(collections.Counter(f"{sc(j)['faults'][0]['type']}/{sc(j)['faults'][0].get('submain')}"
                                                      for j in jobs if j["task"] == "fault_response").items())),
                rotation_events=dict(sorted(collections.Counter(str(sc(j)["irrigation_event"]) for j in jobs if j["task"] == "rotation_plan").items())),
                switch_events=dict(sorted(collections.Counter(str(sc(j)["irrigation_event"]) for j in jobs if j["task"] == "switch").items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-switch", type=int, default=40)
    ap.add_argument("--n-rotation", type=int, default=30)
    ap.add_argument("--n-fault", type=int, default=120)
    ap.add_argument("--out", default=str(ST / "student" / "sets" / "val_big.json"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    S = [json.loads(l) for l in open(ST / "scenarios.jsonl", encoding="utf-8")]
    by_id = {s["id"]: s for s in S}
    used = {j["scenario"] for j in json.load(open(ST / "student" / "sets" / "val_ckpt.json", encoding="utf-8"))["jobs"]}
    if out.exists() and not a.force:
        jobs = json.load(open(out, encoding="utf-8"))["jobs"]
        print(f"已存在 {out}（{len(jobs)} 个作业），不覆盖")
    else:
        rng = np.random.default_rng(SEL_SEED)

        def alive(s):                                                 # 与 distill_gen.run_job 判退化的方式一致
            sc = json.loads(json.dumps(s)); sc["iso_transient"] = True
            return bool(teacher.make_state("fault_response", sc, np.random.default_rng([JOB_SEED, int(s["id"][1:]), 0]))[0].valves)
        nom = [s for s in S if s["split"] == "val" and s["kind"] == "nominal" and s["id"] not in used]
        fau = [s for s in S if s["split"] == "val" and s["kind"] == "fault" and s["id"] not in used and alive(s)]
        by_event = collections.defaultdict(list)
        for s in nom:
            by_event[s["irrigation_event"]].append(s["id"])
        jobs = [dict(job_id=f"ro_{i}_s0", task="rotation_plan", scenario=i, variant={}, sample=0, seed=JOB_SEED) for i in round_robin(by_event, a.n_rotation, rng)]
        sw = round_robin(by_event, a.n_switch, rng)
        pairs = [int(p) for p in rng.permutation(np.resize(np.arange(29), a.n_switch))]
        jobs += [dict(job_id=f"sw_{i}_p{p:02d}_s0", task="switch", scenario=i, variant=dict(pair=p), sample=0, seed=JOB_SEED) for i, p in zip(sw, pairs)]
        for t in ("burst", "leak", "stuck"):
            by_sub = collections.defaultdict(list)
            for s in fau:
                if s["faults"][0]["type"] == t:
                    by_sub[s["faults"][0].get("submain")].append(s["id"])
            jobs += [dict(job_id=f"fa_{i}_t0_s0", task="fault_response", scenario=i, variant=dict(t=0), sample=0, seed=JOB_SEED)
                     for i in round_robin(by_sub, a.n_fault // 3, rng)]
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        json.dump(dict(seed=SEL_SEED, note="val 扩大集（不含 val_ckpt 的场景），v2 与 DPO 配对重评用", jobs=jobs), open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        os.replace(tmp, out)
        print(f"{len(jobs)} 个作业 → {out}")
    # 自检：数量、划分、与 val_ckpt 不重叠、作业名不重复
    n = collections.Counter(j["task"] for j in jobs)
    bad = [j["job_id"] for j in jobs if by_id[j["scenario"]]["split"] != "val" or j["scenario"] in used]
    assert not bad, f"越界作业：{bad[:5]}"
    assert len({j["job_id"] for j in jobs}) == len(jobs), "作业名重复"
    assert (n["switch"], n["rotation_plan"], n["fault_response"]) == (a.n_switch, a.n_rotation, a.n_fault), f"数量不足：{dict(n)}（val 余量不够）"
    for k, v in summary(jobs, by_id).items():
        print(" ", k, v)


if __name__ == "__main__":
    main()
