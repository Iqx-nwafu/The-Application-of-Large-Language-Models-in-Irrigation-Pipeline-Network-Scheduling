"""按 batches.json 启动正式实验批次，并按预实验实测费用估算每批花费。

python runbatch.py --list                       # 列出批次
python runbatch.py --estimate                   # 估算费用（预实验 workflow/prompt_only 每任务平均费用 × 场景数，空闲 / 高峰两种价）
python runbatch.py ds_test kimi_ctrl            # 启动：每批 shards 路后台子进程，日志 results/logs/<tag>.s<i>.log；同一厂商的批次依次跑
python runbatch.py ds_test --dry-run            # 规则桩跑通流程，不调 API

所有子进程都带 --skip-done，掉线或中断后原命令重跑只补缺的。密钥只从环境变量读。
"""
import argparse
import collections
import json
import os
import pathlib
import subprocess
import sys
import threading
import evalrun

HERE = pathlib.Path(__file__).parent
# 批次表：默认 batches.json（主实验）；--config batches_hard.json 用难瞬态子集的批次（单独成表，不改主实验的批次与 analysis 进度口径）
_cfg_name = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else "batches.json"
CFG = json.load(open(HERE / _cfg_name, encoding="utf-8"))
KIND = {"switch": "nominal", "rotation_plan": "nominal", "fault_response": "fault"}


def n_runs(b):
    s = json.load(open(HERE / "sets" / f"{b['set']}.json", encoding="utf-8"))
    return {t: min(len(s[KIND[t]]), b.get("n", 10 ** 6)) * b.get("repeats", 1) for t in b.get("tasks", CFG["tasks"])}


def unit_costs(pattern="results/pre_*.jsonl"):
    """预实验每 (决策器, 模型, 任务) 的平均 token（分项），用来按任一模型的单价重算空闲 / 高峰费用。"""
    import glob
    acc = collections.defaultdict(list)
    for fn in glob.glob(str(HERE / pattern)):
        for l in open(fn, encoding="utf-8"):
            r = json.loads(l)
            if r.get("tokens") and r["maker"] in evalrun.LLM_MAKERS:
                acc[r["maker"], r["model"], r["task"]].append(r["tokens"])
    return {k: {f: sum(t.get(f, 0) for t in v) / len(v) for f in v[0]} | {"n": len(v)} for k, v in acc.items()}


def price(tok, model):
    p = evalrun.PRICES[model]
    return {k: (tok["cache_hit"] * v["input_cache_hit"] + tok["cache_miss"] * v["input_cache_miss"]
                + tok.get("cache_write", 0) * v.get("cache_write", 0) + tok["completion"] * v["output"]) / 1e6
            for k, v in p.items() if isinstance(v, dict)}


def estimate():
    U = unit_costs()
    tot = collections.Counter()
    print(f"{'批次':<16}{'模型':<18}{'次数':>6}{'空闲 元':>10}{'高峰 元':>10}  依据")
    for b in CFG["batches"]:
        if b.get("local"):
            continue
        maker = b["maker"][0]
        proxy = b.get("token_proxy", b["model"])
        runs, off, peak, src = n_runs(b), 0.0, 0.0, []
        for t, n in runs.items():
            tok = U.get((maker, proxy, t)) or U.get(("workflow", proxy, t)) or U.get(("workflow", "deepseek-v4-pro", t))
            c = price(tok, b["model"])
            off += n * c["offpeak"]; peak += n * c["peak"]
            src.append(f"{t[:6]} n={tok['n']}")
        tot["offpeak"] += off; tot["peak"] += peak
        print(f"{b['name']:<16}{b['model']:<18}{sum(runs.values()):>6}{off:>10.1f}{peak:>10.1f}  {proxy} 预实验 token（{', '.join(src)}）")
    print(f"{'合计':<34}{'':>6}{tot['offpeak']:>10.1f}{tot['peak']:>10.1f}")


def cmd(b, i, dry):
    c = [sys.executable, "evalrun.py", "--maker", *b["maker"], "--task", *b.get("tasks", CFG["tasks"]), "--set", b["set"], "--n", str(b.get("n", 10 ** 6)),
         "--repeats", str(b.get("repeats", 1)), "--tag", b["tag"], "--shard", f"{i}/{b['shards']}", "--skip-done"]
    if not b.get("local"):
        c += ["--provider", b["provider"], "--model", b["model"], "--ablation", b.get("ablation", "none")]
    c += (["--outdir", b["outdir"]] if b.get("outdir") else []) + (["--safety"] if b.get("safety") else [])
    return c + (["--dry-run"] if dry else [])


def run_batch(b, dry):
    (HERE / "results" / "logs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    procs = []
    for i in range(b["shards"]):
        log = open(HERE / "results" / "logs" / f"{b['tag']}.s{i}.log", "a", encoding="utf-8")
        procs.append(subprocess.Popen(cmd(b, i, dry), cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env=env,
                                      creationflags=subprocess.CREATE_NO_WINDOW))   # 各自隐藏控制台，不吃别处的 Ctrl+C
    codes = [p.wait() for p in procs]
    print(f"[{b['name']}] 结束，退出码 {codes}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--config", default="batches.json", help="批次表文件（模块加载时已读取）")
    a = ap.parse_args()
    by = {b["name"]: b for b in CFG["batches"]}
    if a.list:
        for b in CFG["batches"]:
            print(f"{b['name']:<16}{b.get('model', 'local'):<18}{b['set']:<12}{b['shards']} 路 {b['tag']}  {b.get('_why', '')}")
    if a.estimate:
        estimate()
    if a.list or a.estimate:                                     # 只看清单/估价，不启动（2026-10-31 误启事故）
        return
    queues = collections.defaultdict(list)                       # 同一厂商排队，不同厂商并行
    for n in a.names:
        queues[by[n].get("provider", "local")].append(by[n])
    threads = [threading.Thread(target=lambda q=q: [run_batch(b, a.dry_run) for b in q]) for q in queues.values()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
