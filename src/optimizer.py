"""A8 优化器基线：NSGA-II 做轮灌分组（不用 GJO），也是工作流里的 run_optimizer 工具和兜底层的后备。

编码：每个可用支管阀一个组标签 ∈ [0, K_MAX)，空标签丢弃；组序按（分干管, 节点）字典序。
灌水时长由 evaluate.schedule 按实际流量和需水确定，所以搜索只在分组上。
目标：缺水量、CV_q、轮灌总时长、超灌水量；约束：evaluate.py 的全部稳态约束（G ≤ 0）。
选解按方案 A（scoring.py）：可行 → 缺水字典序 → 熵权 TOPSIS 贴近度最大；并报告权重敏感性。

python optimizer.py [场景 id]      # 默认无故障、定额 25 m3/亩
python optimizer.py reference      # 用 4 个标称场景的前沿建 TOPSIS 参考集（熵权 + 范围）
"""
import contextlib
import json
try:
    import msvcrt
except ImportError:                                               # Linux（云上）：用 fcntl.flock 实现同样的非阻塞槽锁
    import fcntl
    msvcrt = None
import os
import pathlib
import sys
import time
import numpy as np
from multiprocessing import Pool
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.mutation import Mutation
from pymoo.core.repair import Repair
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.optimize import minimize
import evaluate as ev
import scoring

HERE = pathlib.Path(__file__).parent
K_MAX, G_MAX = 60, 8
CONSTRAINTS = ["Q_hub", "H_min", "v_max", "PN_PE", "PN_PVC", "T_window", "group_size"]


def decode(ctx, labels):
    groups = {}
    for v, l in zip(ctx.valves, labels):
        groups.setdefault(int(l), []).append(v)
    return sorted(groups.values(), key=lambda g: min(g))


_ctx = None


def _init(scenario, apply_faults, exclude=(), t_win=None):
    """exclude: 本次不参与分组的支管阀（已灌完的）；t_win: 剩余时间窗（故障处置时）。"""
    global _ctx
    _ctx = ev.Context(scenario, apply_faults)
    _ctx.valves = [v for v in _ctx.valves if v not in set(exclude)]
    if t_win:
        _ctx.t_win = t_win


def _eval(labels):
    r = ev.evaluate(_ctx, decode(_ctx, labels))
    o, vi = r["objectives"], r["violations"]
    return [o["deficit_m3"], o["cv_q"], o["T_total_h"], o["over_m3"]], [vi.get(c, 0.0) for c in CONSTRAINTS]


class Rotation(Problem):
    def __init__(self, n, pool):
        super().__init__(n_var=n, n_obj=4, n_ieq_constr=len(CONSTRAINTS), xl=0, xu=K_MAX - 1, vtype=int)
        self.pool = pool

    def _evaluate(self, X, out, *args, **kwargs):
        res = self.pool.map(_eval, [x.astype(int) for x in X])
        out["F"] = np.array([r[0] for r in res])
        out["G"] = np.array([r[1] for r in res])


class SizeRepair(Repair):
    """每组不超过 G_MAX 个阀：多出的阀挪到空标签或最小的组。"""
    def _do(self, problem, X, **kwargs):
        X = X.astype(int)
        for x in X:
            cnt = np.bincount(x, minlength=K_MAX)
            for g in np.where(cnt > G_MAX)[0]:
                for i in np.where(x == g)[0][G_MAX:]:
                    cnt = np.bincount(x, minlength=K_MAX)
                    tgt = np.argmin(np.where(cnt == 0, G_MAX, cnt))   # 优先放进最小的非空组，满了再开新组
                    x[i] = tgt
        return X


class Seeded(Sampling):
    """一半随机（组数 20–40），一半跨分干管交错的启发式种子：每组取不同分干管上的支管。"""
    def __init__(self, valves):
        super().__init__(); self.valves = valves

    def _do(self, problem, n, **kwargs):
        rng = np.random.default_rng(0)
        X = []
        for i in range(n):
            if i % 2 == 0:
                k = rng.integers(20, 41)
                X.append(rng.integers(0, k, problem.n_var))
            else:
                per = int(rng.choice([3, 4]))
                order = sorted(range(len(self.valves)), key=lambda j: (self.valves[j][2:], self.valves[j][1]))
                x = np.zeros(problem.n_var, int)
                for rank, j in enumerate(order):
                    x[j] = rank // per
                X.append(np.minimum(x, K_MAX - 1))
        return np.array(X)


class Reassign(Mutation):
    """随机挑 1–3 个阀换到别的组（含新开一组），或交换两个阀的组。"""
    def _do(self, problem, X, **kwargs):
        rng = np.random.default_rng()
        X = X.astype(int).copy()
        for x in X:
            for _ in range(rng.integers(1, 4)):
                if rng.uniform() < 0.5:
                    x[rng.integers(len(x))] = rng.integers(K_MAX)
                else:
                    i, j = rng.integers(len(x), size=2); x[i], x[j] = x[j], x[i]
        return X


def _cfg(name, default):
    """results/<name>.txt 里的一个整数，可在运行中改（压内存 / 调并发用）；没有就用默认。"""
    try:
        return int((HERE / "results" / f"{name}.txt").read_text().strip())
    except (OSError, ValueError):
        return default


@contextlib.contextmanager
def _machine_lock():
    """全机同时最多开 ga_slots 个 GA 进程池（默认 3 × 8 路 = 24 核）：多个评测进程各自起 22 路池曾把页面文件撑爆（2026-10-28 03:07 事故）。
    每个槽一个锁文件，非阻塞轮询抢任意空槽；返回等槽秒数。"""
    t = time.time()
    files = [open(HERE / "results" / f".ga_pool.lock.{i}", "a+b") for i in range(_cfg("ga_slots", 3))]
    got = None
    while got is None:
        for f in files:
            try:
                f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1) if msvcrt else fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                got = f
                break
            except OSError:
                continue
        if got is None:
            time.sleep(2)
    try:
        yield round(time.time() - t, 1)
    finally:
        got.seek(0); msvcrt.locking(got.fileno(), msvcrt.LK_UNLCK, 1) if msvcrt else fcntl.flock(got.fileno(), fcntl.LOCK_UN)
        for f in files:
            f.close()


def run_optimizer(scenario=None, apply_faults=True, exclude=(), t_win=None, pop=120, gens=150, workers=22, seed=1, verbose=False):
    _init(scenario, apply_faults, exclude, t_win)
    ctx = _ctx
    workers = _cfg("ga_workers", int(os.environ.get("IRRIG_GA_WORKERS", workers)))   # 文件优先于环境变量，运行中可改
    with _machine_lock() as lock_wait_s, Pool(workers, initializer=_init, initargs=(scenario, apply_faults, tuple(exclude), t_win)) as pool:
        t0 = time.time()                                          # 不计排队等锁的时间
        prob = Rotation(len(ctx.valves), pool)
        alg = NSGA2(pop_size=pop, sampling=Seeded(ctx.valves), crossover=UniformCrossover(prob=0.7),
                    mutation=Reassign(), repair=SizeRepair(), eliminate_duplicates=False)
        res = minimize(prob, alg, ("n_gen", gens), seed=seed, verbose=verbose)
        cpu_s = round(time.time() - t0, 1)
    X =np.atleast_2d(res.X) if res.X is not None else np.atleast_2d(res.pop.get("X"))
    cands = [ev.evaluate(ctx, decode(ctx, x)) for x in X]
    i, sel = scoring.select(cands)
    best = cands[i]
    return dict(best=best, rotation_plan=ev.to_rotation_plan(best), n_pareto=len(cands), n_feasible=sel["n_feasible"],
                selection=sel, cpu_s=cpu_s, lock_wait_s=lock_wait_s, n_eval=pop * gens,
                front=[c["objectives"] for c in cands if c["feasible"]])


def build_reference(workers=22):
    """标称场景（定额 20/23/25/25.7，δ=0.1，train）各跑一次，取字典序可行解合并建 TOPSIS 参考集。"""
    S = [json.loads(l) for l in open(HERE / "scenarios.jsonl", encoding="utf-8")]
    picks, fronts = [], []
    for q in (20, 23, 25, 25.7):
        sc = next(s for s in S if s["kind"] == "nominal" and s["split"] == "train" and s["quota_m3_per_mu"] == q
                  and s["perturbation"]["delta"] == 0.1)
        r = run_optimizer(sc, workers=workers)
        d = [f["deficit_m3"] for f in r["front"]]
        tol = scoring.d_tol(r["best"]["D_total"])
        F = [[f[k] for k in scoring.OBJ] for f in r["front"] if f["deficit_m3"] <= min(d) + tol]
        fronts.append(F); picks.append(sc["id"])
        print(f"{sc['id']} 定额 {q}: 可行 {len(r['front'])}，字典序 {len(F)}，缺水最小 {min(d):.1f}，{r['cpu_s']} s")
    ref = scoring.build_reference(fronts, note=f"标称场景 {picks} 的 NSGA-II 字典序可行解合并，{time.strftime('%Y-%m-%d')}")
    print("参考集：", ref)
    return ref


if __name__ == "__main__":
    import network as net
    if sys.argv[1:] == ["reference"]:
        build_reference(); sys.exit()
    sc = None
    if len(sys.argv) > 1:
        sc = next(json.loads(l) for l in open(HERE / "scenarios.jsonl", encoding="utf-8") if json.loads(l)["id"] == sys.argv[1])
    out = run_optimizer(sc, verbose=False)
    b, sel = out["best"], out["selection"]
    print(f"NSGA-II {out['n_eval']} 次评价，{out['cpu_s']} s；Pareto {out['n_pareto']} 个，可行 {sel['n_feasible']} 个，字典序 {sel['n_lexi']} 个")
    print("选解:", b["objectives"], "C* =", b["score"], "越限", b["violations"])
    print("权重", sel["weights"], "参考集" if sel["reference"] else "本前沿熵权", "敏感性稳定" if sel["sensitivity"]["stable"] else f"敏感性换解 {sel['sensitivity']['chosen']}")
    print("组数", len(b["per_group"]), "每组阀数", [len(g["valves"]) for g in b["per_group"]],
          "节流阀", sum(len(g["openings"]) for g in b["per_group"]))
    base = ev.evaluate(ev.Context(sc), net.rotation_groups(), durations=[4.0] * 30)
    print("基线轮灌表:", base["objectives"], "C* =", base["score"], "越限", base["violations"])
    name = sys.argv[1] if len(sys.argv) > 1 else "nominal"
    json.dump(out, open(HERE / f"opt_{name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
