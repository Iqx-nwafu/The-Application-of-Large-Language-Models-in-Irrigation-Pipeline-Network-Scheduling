"""正式实验的固定场景集（第一轮前生成一次，之后不再改）→ sets/<名>.json = {"nominal": [id…], "fault": [id…]}

测试集按层排成一个"系统抽样序"：每层内随机打乱，第 j 个样本的键 = (j + u) / 层大小（u 每层一个随机偏移），全体按键排序。
于是任意前缀在各层上都近似按比例分配，各子集取同一序列的前缀，天然嵌套：
  gpt20 ⊂ repro30 ⊂ control40 ⊂ control60 ⊂ ablation100 ⊂ test（全量 182 + 186）
分层：标称场景按灌水次序（表 3.2-1 的 12 次，定额不同）；故障场景粗层按 故障类型 × 管道级别（卡死阀一类），
粗层内再按 卡死模式 × 分干管位置（1–3 / 4–6 / 7–9）做系统抽样。
分布外 ood150 = 标称 38 + 故障 112，同法在 ood 场景上按 ood_tags × 故障类型分层。

python make_sets.py            # 写 sets/*.json 并打印各层计数
"""
import collections
import json
import pathlib
import numpy as np

HERE = pathlib.Path(__file__).parent
SEED = 20261028
SIZES = {"gpt20": (10, 10), "repro30": (15, 15), "control40": (20, 20), "control60": (30, 30), "ablation100": (50, 50)}


def stratum(sc):
    """(粗层, 细层)：粗层决定比例分配，细层在粗层内部再做一次系统抽样，保证小前缀也覆盖位置与卡死模式。"""
    if sc["kind"] == "nominal":
        return ("ev", sc["irrigation_event"]) + tuple(sc["ood_tags"]), ()
    f = sc["faults"][0]
    pos = (f["submain"] - 1) // 3 if f.get("submain") else "main"
    return (f["type"], f.get("pipe_class") or "valve") + tuple(sc["ood_tags"]), (f.get("mode", ""), pos)


def _sys_keys(items, key, rng):
    groups = collections.defaultdict(list)
    for s in items:
        groups[key(s)].append(s)
    keyed = []
    for k in sorted(groups, key=str):
        g = groups[k]
        rng.shuffle(g)
        u = rng.random()
        keyed += [((j + u) / len(g), s) for j, s in enumerate(g)]
    return [s for _, s in sorted(keyed, key=lambda t: (t[0], t[1]["id"]))]


def systematic_order(scs, rng):
    groups = collections.defaultdict(list)
    for s in scs:
        groups[stratum(s)[0]].append(s)
    keyed = []
    for k in sorted(groups, key=str):
        g = _sys_keys(groups[k], lambda s: stratum(s)[1], rng)        # 粗层内先按细层排成系统抽样序
        u = rng.random()
        keyed += [((j + u) / len(g), s["id"]) for j, s in enumerate(g)]
    return [i for _, i in sorted(keyed)]


def main():
    S = [json.loads(l) for l in open(HERE / "scenarios.jsonl", encoding="utf-8")]
    rng = np.random.default_rng(SEED)
    order = {(sp, kind): systematic_order([s for s in S if s["split"] == sp and s["kind"] == kind], rng)
             for sp in ("test", "ood") for kind in ("nominal", "fault")}
    sets = {"test": dict(nominal=order["test", "nominal"], fault=order["test", "fault"]),
            "ood150": dict(nominal=order["ood", "nominal"][:38], fault=order["ood", "fault"][:112])}
    for name, (nn, nf) in SIZES.items():
        sets[name] = dict(nominal=order["test", "nominal"][:nn], fault=order["test", "fault"][:nf])
    (HERE / "sets").mkdir(exist_ok=True)
    by_id = {s["id"]: s for s in S}
    for name, d in sets.items():
        json.dump(dict(d, _note=f"make_sets.py seed {SEED}；前缀嵌套，勿手改"), open(HERE / "sets" / f"{name}.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        cnt = collections.Counter(stratum(by_id[i])[0][:2] for k in ("nominal", "fault") for i in d[k])
        print(f"{name:<12} 标称 {len(d['nominal']):>3} 故障 {len(d['fault']):>3} ", dict(sorted(cnt.items(), key=str)))


if __name__ == "__main__":
    main()
