"""等蒸馏进程全部退出后出最终候选池（会话断了也照常执行，用 WMI + pythonw 起）：
to_sft 不限长 + --repair + 每状态最多 99 条（同 student/README 的 pool 命令）→ <out>；再按 2.25 字节/token 估 ≤24k / ≤32k 的可用数，写 <out>/summary.json。

pythonw distill/final_pool.py <work>/pool_b2
"""
import collections
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
out = sys.argv[1]
log = open(HERE / "logs" / "final_pool.log", "a", encoding="utf-8")


def busy():
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Where-Object { $_.CommandLine -like '*distill_gen.py*' "
                        "-or $_.CommandLine -like '*launch_distill.py*' } | Measure-Object).Count"],
                       capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return r.stdout.strip() != "0"


while busy():
    time.sleep(120)
log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} 蒸馏进程已全部退出，开始出 {out}\n"); log.flush()
env = dict(os.environ, PYTHONIOENCODING="utf-8", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
py = pathlib.Path(sys.executable).with_name("python.exe")
subprocess.run([str(py), "-W", "ignore", str(HERE / "to_sft.py"), "--src", str(HERE / "raw_flash"), str(HERE / "raw"), "--repair",
                "--max-tokens", "100000000", "--max-per-state", "99", "--out", out], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
c = collections.defaultdict(collections.Counter)
for l in open(pathlib.Path(out) / "index.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if not r["keep"]:
        continue
    k = r["task"] + ("_stress" if r["job_id"].startswith("sw_T") else "")
    n = r["n_tokens"] or 0
    c[k]["unlimited"] += 1
    c[k]["le32k"] += n <= 32768
    c[k]["le24k"] += n <= 24576
    c[k][r["teacher"]] += 1
summary = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), by_task={k: dict(v) for k, v in c.items()},
               in_dist={t: sum(v[t] for k, v in c.items() if "stress" not in k) for t in ("unlimited", "le32k", "le24k")})
json.dump(summary, open(pathlib.Path(out) / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} 完成：{json.dumps(summary['in_dist'], ensure_ascii=False)}\n")
log.close()
