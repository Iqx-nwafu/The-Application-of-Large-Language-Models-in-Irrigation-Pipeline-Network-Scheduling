"""本机部署评测驱动（v3 = DPO a s00085；RTX 5060 Laptop 8 GB + llama.cpp b12032）。irrig 环境：
  python run_local_v3.py smoke     前台冒烟：Q8_0 起服务，跑 student/sets/_smoke.json 的 3 条，末行 SMOKE_OK / SMOKE_FAILED
  python run_local_v3.py launch    用 WMI + pythonw 脱离会话起 chain，立即返回
  python run_local_v3.py chain     依次跑各阶段，单路串行；已完成的作业跳过，可随时停、重新 launch 即续跑
  python run_local_v3.py status    打印当前状态（同 status\\latest.txt）
暂停：在本目录放一个空文件 STOP（当前这一条作业作废、停服务后退出）；删掉 STOP 再 launch 即续跑。
阶段（都用 eval_student.py 的 1 路串行口径，温度 0、32k 上下文、每轮工具上限 20）：
  repro      正式时延与可重复性：repro30 的 45 个状态 × 10 次，状态与教师 repro_ds 逐条相同；Q8_0 权重、f16 KV（部署配置）
  prec_*     三档精度对比：val_ckpt 60 条，Q8_0 / bf16 / Q4_K_M 权重，KV 统一 q8_0（bf16 + 32k + f16 KV 放不进 8 GB）
  cpu        纯 CPU 探针：Q8_0、-ngl 0，9 条
每个阶段旁边开 gpu_monitor.py（nvidia-smi 0.5 s + WDDM 专用 / 共享显存 1 s）→ mon\\<阶段>_smi.csv、_wddm.csv。
状态每 30 s 写 status\\latest.txt，事件写 status\\events.log，全部完成写 status\\chain.finished。
llama-server 中途断线：停评测、删掉该阶段的"连接错误"记录后重起服务续跑（每阶段最多 3 次失败）。
系统挂起（合盖 / 睡眠）：两次巡检间隔超过 120 s 即判挂起，停评测、删掉挂起前后写出的记录（墙钟被挂起污染）后续跑，不计失败。
每次停服务后把 llama-server 日志另存一份到 logs\\server_<阶段>_<时间>.log。"""
import ctypes
import json
import os
import pathlib
import re
import shutil
import statistics as st
import subprocess
import sys
import time
import urllib.request

NT = os.name == "nt"
T = os.environ.get("LOCAL_V3_TEST")                                   # 沙箱测试：路径改到这个目录下
ROOT = pathlib.Path(__file__).resolve().parents[2]
STAGEA = pathlib.Path(T, "stageA") if T else ROOT / "src"
W = pathlib.Path(T, "local_v3") if T else pathlib.Path(os.environ.get("DEPLOY_WORK", ROOT / "work" / "deploy"))
G = pathlib.Path(T, "gguf") if T else pathlib.Path(os.environ.get("V3_GGUF_DIR", ROOT / "models" / "gguf" / "v3_dpo_a_s00085"))
ST, OUT = STAGEA / "student", STAGEA / "results" / "student"
PORT = 8091
PY = str(pathlib.Path(sys.executable).with_name("python.exe")) if NT else sys.executable
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
TICK, GAP = (0.2, 3) if T else (15, 120)
SERVER_LOG = pathlib.Path(T, "server_8091.log") if T else pathlib.Path(os.environ.get("LLAMA_LOG_DIR", W)) / "server_8091.log"
REF_S = 150                                                           # 一个 GA 参照的估计用时（分组 / 故障状态各算一次后缓存）
STAGES = [
    dict(name="repro", set="repro30_v3", tag="v3_s00085_q8_repro", gguf="model-Q8_0.gguf", kv="f16", ngl=99, threads=4),
    dict(name="prec_q8", set="val_ckpt", tag="v3_s00085_q8_kvq8", gguf="model-Q8_0.gguf", kv="q8_0", ngl=99, threads=4),
    dict(name="prec_bf16", set="val_ckpt", tag="v3_s00085_bf16_kvq8", gguf="model-bf16.gguf", kv="q8_0", ngl=99, threads=4),
    dict(name="prec_q4", set="val_ckpt", tag="v3_s00085_q4km_kvq8", gguf="model-Q4_K_M.gguf", kv="q8_0", ngl=99, threads=4),
    dict(name="cpu", set="cpu_probe_v3", tag="v3_s00085_q8_cpu", gguf="model-Q8_0.gguf", kv="f16", ngl=0, threads=8),
]
PRIOR_S = dict(switch=35, rotation_plan=280, fault_response=310)      # 没有实测时的每条用时估计
TASK_ZH = (("switch", "切换"), ("rotation_plan", "分组"), ("fault_response", "故障"))
CONN = re.compile(r"APIConnectionError|Connection error|refused|RemoteDisconnected|Max retries")   # 不含超时：服务活着但单次请求超 20 min 是真实结果


DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # 探本机服务不走系统代理


class Paused(Exception):
    pass


def log(msg):
    line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
    with open(W / "status" / "events.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def run(cmd, timeout=60, **kw):
    try:
        r = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL, timeout=timeout, creationflags=NOWIN, env=ENV, **kw)
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, f"{type(e).__name__}: {e}"
    out = r.stdout + r.stderr
    try:
        return r.returncode, out.decode("utf-8").strip()
    except UnicodeDecodeError:                                          # PowerShell / powercfg 的中文输出是 GBK
        return r.returncode, out.decode("gbk", "replace").strip()


def kill(p):
    if p is None or p.poll() is not None:
        return
    if NT:
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True, stdin=subprocess.DEVNULL, creationflags=NOWIN)
    else:
        p.kill()
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def jobs_of(s):
    p = ST / "sets" / f"{s['set']}.json"
    return json.load(open(p, encoding="utf-8"))["jobs"] if p.exists() else []


def rec_path(s, j):
    return OUT / s["tag"] / j["task"] / f"{j['job_id']}.json"


CACHE = {}


def info(p):
    """一条结果记录的要点（读一次后缓存）；文件不在或还没写完返回 None。"""
    if p in CACHE:
        return CACHE[p]
    if not p.exists():
        return None
    try:
        r = json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        return None
    o, lat = r.get("outcome") or {}, r.get("latency") or {}
    dec = [u["timings"]["predicted_per_second"] for m in r.get("messages") or [] if m.get("role") == "assistant"
           for u in [m.get("_usage") or {}] if (u.get("timings") or {}).get("predicted_per_second")]
    CACHE[p] = dict(task=r["task"], status=r.get("status"), ok=r.get("status") == "ok", bf=r.get("status") == "ok" and bool(o.get("feasible_before_fallback")),
                    wall=lat.get("wall_s"), model=lat.get("model_s"), inwin=bool(lat.get("in_window")), err=str(r.get("error", "")),
                    tools=o.get("tool_calls"), rounds=o.get("n_rounds"), decode=st.median(dec) if dec else None)
    return CACHE[p]


def done_of(s, jobs):
    return [x for x in (info(rec_path(s, j)) for j in jobs) if x]


def clean_conn_errors(s, jobs):
    """llama-server 断线后 worker 会把后面的作业秒记为连接错误：删掉这些记录，重跑时补上。"""
    n = 0
    for j in jobs:
        p, x = rec_path(s, j), info(rec_path(s, j))
        if x and x["status"] == "error" and CONN.search(x["err"]):
            os.remove(p)
            CACHE.pop(p, None)
            n += 1
    if n:
        log(f"[{s['name']}] 删掉 {n} 条连接错误记录，重跑时补上")


def serve(action, s=None):
    cmd = [PY, str(ST / "serve.py"), action]
    if s:
        cmd += [str(G / s["gguf"]), "--kv", s["kv"], "--ngl", str(s["ngl"]), "--threads", str(s["threads"])]
    return run(cmd + ["--port", str(PORT)], timeout=300)


def healthy():
    try:
        return json.load(DIRECT.open(f"http://127.0.0.1:{PORT}/health", timeout=10)).get("status") == "ok"
    except (OSError, ValueError):
        return False


def gpu_line():
    rc, out = run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader"], timeout=20)
    return out.splitlines()[0] if rc == 0 and out else "nvidia-smi 无输出"


def med(v):
    v = [x for x in v if x is not None]
    return f"{st.median(v):.0f}" if v else "-"


def state_key(j):                                                     # 同 eval_student.state_key
    v = j["variant"]
    return f"{j['task']}|{j['scenario']}|{v.get('pair', v.get('t', ''))}"


def stage_line(s, ref):
    jobs = jobs_of(s)
    d = done_of(s, jobs)
    todo = {state_key(j) for j in jobs if j["task"] != "switch" and not info(rec_path(s, j))}
    parts, left = [], REF_S * sum(k not in ref for k in todo)
    for t, zh in TASK_ZH:
        n = sum(j["task"] == t for j in jobs)
        if not n:
            continue
        x = [r for r in d if r["task"] == t]
        w = [r["wall"] for r in x if r["wall"] is not None]
        left += (n - len(x)) * (sum(w) / len(w) + 15 if w else PRIOR_S[t] * (5 if s["ngl"] == 0 else 1))
        parts.append(f"{zh} {len(x)}/{n}（可行 {sum(r['bf'] for r in x)}，窗口内 {sum(r['inwin'] for r in x)}，出错 {sum(not r['ok'] for r in x)}，"
                     f"墙钟中位 {med(r['wall'] for r in x)} s，模型中位 {med(r['model'] for r in x)} s）")
    return f"{s['name']}：完成 {len(d)}/{len(jobs)}；" + "；".join(parts), left, len(d), len(jobs)


def write_status(state, cur=None, note=""):
    L, total_left, ref = [], 0.0, {}
    try:
        ref = json.load(open(OUT / "opt_ref.json", encoding="utf-8"))
    except (OSError, ValueError):
        pass
    for s in STAGES:
        line, left, _, _ = stage_line(s, ref)
        L.append(line)
        total_left += left
    eta = time.strftime("%m-%d %H:%M", time.localtime(time.time() + total_left))
    head = [f"时间 {time.strftime('%m-%d %H:%M:%S')}（本机时间）  状态：{state}  当前阶段：{cur['name'] if cur else '-'}",
            f"估计还需 {total_left / 3600:.1f} h，约 {eta} 结束（按已完成作业的平均用时估）", f"显卡（已用显存, 利用率, 温度, 功耗）：{gpu_line()}"]
    if note:
        head.append(note)
    tail = []
    if cur:
        lp = W / "logs" / f"{cur['tag']}.log"
        if lp.exists():
            with open(lp, encoding="utf-8", errors="replace") as f:
                last = f.read()[-1200:].strip().splitlines()
            tail = ["最近的评测日志：" + (last[-1][:300] if last else "")]
    tmp = W / "status" / "latest.tmp"
    tmp.write_text("\n".join(head + L + tail) + "\n", encoding="utf-8")
    os.replace(tmp, W / "status" / "latest.txt")


def keep_server_log(s):
    if SERVER_LOG.exists():
        shutil.copy(SERVER_LOG, W / "logs" / f"server_{s['name']}_{time.strftime('%m%d%H%M%S')}.log")


def run_stage(s, stoppable=True):
    jobs = jobs_of(s)
    assert jobs, f"场景集 {s['set']} 不存在或为空"
    assert (G / s["gguf"]).exists(), f"缺 {G / s['gguf']}"
    fails = 0
    while fails < 3:
        clean_conn_errors(s, jobs)
        if len(done_of(s, jobs)) == len(jobs):
            return True
        serve("stop")
        rc, out = serve("start", s)
        log(f"[{s['name']}] 起服务 rc={rc}：{out[-200:]}")
        if rc != 0 or not healthy():
            fails += 1
            serve("stop")
            time.sleep(TICK)
            continue
        mon = subprocess.Popen([PY, str(ST / "gpu_monitor.py"), str(W / "mon" / s["name"])], stdin=subprocess.DEVNULL, creationflags=NOWIN, env=ENV)
        lf = open(W / "logs" / f"{s['tag']}.log", "a", encoding="utf-8")
        p = subprocess.Popen([PY, str(ST / "eval_student.py"), "--set", s["set"], "--tag", s["tag"], "--port", str(PORT), "--shard", "0/1"],
                             cwd=STAGEA, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT, env=ENV, creationflags=NOWIN)
        bad, down, gap, last = 0, False, None, time.time()
        try:
            while p.poll() is None:
                time.sleep(TICK)
                now = time.time()
                if now - last > GAP:                                    # 巡检间隔远超设定：系统挂起过
                    gap = (last, now)
                    break
                last = now
                write_status("运行中", s)
                if stoppable and (W / "STOP").exists():
                    raise Paused
                bad = 0 if healthy() else bad + 1
                if bad >= 3:                                            # 连续三次探不到服务（约 45 s）：判断线
                    down = True
                    break
        finally:
            kill(p)
            kill(mon)
            lf.close()
            serve("stop")
            keep_server_log(s)
        if gap:
            hit = [j["job_id"] for j in jobs if rec_path(s, j).exists() and rec_path(s, j).stat().st_mtime >= gap[0]]
            for j in jobs:
                if j["job_id"] in hit:
                    os.remove(rec_path(s, j))
                    CACHE.pop(rec_path(s, j), None)
            log(f"[{s['name']}] 系统挂起约 {gap[1] - gap[0]:.0f} s：停评测，删掉挂起前后写出的记录 {hit}，续跑（不计失败）")
            continue
        n = len(done_of(s, jobs))
        log(f"[{s['name']}] 评测进程结束 rc={p.returncode}{'（服务断线）' if down else ''}，完成 {n}/{len(jobs)}")
        clean_conn_errors(s, jobs)
        fails += len(done_of(s, jobs)) < len(jobs)
    return False


def env_snapshot():
    q = dict(gpu=["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader"],
             cpu=["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Processor | Select-Object -First 1 Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress"],
             ram_gb=["powershell", "-NoProfile", "-Command", "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)"],
             battery=["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Battery | Select-Object -First 1 BatteryStatus,EstimatedChargeRemaining | ConvertTo-Json -Compress"],
             power_plan=["powercfg", "/getactivescheme"], os=["cmd", "/c", "ver"])
    snap = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), python=sys.version.split()[0])
    for k, cmd in q.items():
        snap[k] = run(cmd, timeout=40)[1] if NT else "(非 Windows)"
    for f in ("ga_slots.txt", "ga_workers.txt"):
        p = STAGEA / "results" / f
        snap[f] = p.read_text().strip() if p.exists() else None
    json.dump(snap, open(W / "status" / "env.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    on_battery = bool(re.search(r'"BatteryStatus":1[,}]', str(snap.get("battery", "")).replace(" ", "")))   # 1 = 放电中
    return snap, on_battery


def single_instance():
    f = open(W / "status" / "chain.lock", "a+")
    f.seek(0)
    try:
        if NT:
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return None
    return f


def chain():
    sys.stdout = sys.stderr = open(W / "logs" / "chain.out", "a", encoding="utf-8", buffering=1)
    lock = single_instance()
    if lock is None:
        log("已有一个 chain 在跑，本次不重复启动")
        return
    if NT:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)      # 运行期间不因空闲而睡眠（合盖仍会睡）
    (W / "status" / "chain.finished").unlink(missing_ok=True)
    (W / "status" / "chain.paused").unlink(missing_ok=True)
    if (W / "STOP").exists():
        log("目录里有 STOP 文件，不启动；删掉它再 launch")
        (W / "status" / "chain.paused").write_text("STOP 文件在，未启动\n", encoding="utf-8")
        return
    snap, on_battery = env_snapshot()
    log(f"chain 启动 pid {os.getpid()}；显卡 {snap['gpu']}；电源 {snap['battery']}{'  ！！正在用电池' if on_battery else ''}")
    failed, cur = [], None
    try:
        for cur in STAGES:
            ok = run_stage(cur)
            log(f"[{cur['name']}] {'完成' if ok else '未跑满（已失败 3 次），继续下一阶段'}")
            if not ok:
                failed.append(cur["name"])
    except Paused:
        log(f"[{cur['name']}] 收到 STOP，已停评测与服务；删掉 STOP 再 launch 即续跑")
        (W / "status" / "chain.paused").write_text(time.strftime("%m-%d %H:%M:%S") + "\n", encoding="utf-8")
        write_status("已暂停（STOP）", cur)
        return
    except Exception as e:                                              # 驱动自身出错：留下原因，便于远程看
        log(f"chain 出错退出：{type(e).__name__}: {e}")
        write_status(f"出错退出：{type(e).__name__}: {e}", cur)
        raise
    tags = [s["tag"] for s in STAGES if done_of(s, jobs_of(s))]
    rc, out = run([PY, str(ST / "eval_student.py"), "--summary", *tags, "--csv", str(W / "out" / "summary_local_v3.csv")], timeout=1800, cwd=STAGEA)
    (W / "out" / "summary_local_v3.tsv").write_text(out + "\n", encoding="utf-8")
    write_status("全部结束" + (f"（未跑满的阶段：{'、'.join(failed)}）" if failed else ""))
    (W / "status" / "chain.finished").write_text(time.strftime("%m-%d %H:%M:%S") + (" failed=" + ",".join(failed) if failed else "") + "\n", encoding="utf-8")
    log("chain 全部结束" + (f"；未跑满：{failed}" if failed else ""))


def smoke():
    s = dict(name="smoke", set="_smoke", tag="v3_smoke_q8_" + time.strftime("%m%d%H%M"), gguf="model-Q8_0.gguf", kv="f16", ngl=99, threads=4)
    snap, on_battery = env_snapshot()
    print("环境：", json.dumps(snap, ensure_ascii=False))
    idle = gpu_line()
    ok = run_stage(s, stoppable=False)
    jobs = jobs_of(s)
    d = done_of(s, jobs)
    for j in jobs:
        print(j["job_id"], info(rec_path(s, j)))
    peak = 0
    smi = W / "mon" / "smoke_smi.csv"
    if smi.exists():
        peak = max([int(m.group(1)) for m in re.finditer(r"(\d+) MiB", smi.read_text(encoding="utf-8", errors="replace"))] or [0])
    sw = [x for x in d if x["task"] == "switch"]
    dec = [x["decode"] for x in d if x["decode"]]
    checks = dict(all_written=ok and len(d) == len(jobs), switch_feasible=bool(sw) and all(x["bf"] for x in sw),
                  decode_ok=bool(dec) and st.median(dec) >= 40, vram_ok=0 < peak < 7900, on_ac=not on_battery)
    print(f"空闲时显卡：{idle}；评测中已用显存峰值 {peak} MiB；解码中位 {med(dec)} tok/s")
    print("检查：", checks)
    print("SMOKE_OK" if all(checks.values()) else "SMOKE_FAILED")


def launch():
    pyw = pathlib.Path(sys.executable).with_name("pythonw.exe")
    exe, me = (pyw if pyw.exists() else pathlib.Path(PY)), pathlib.Path(__file__).resolve()
    assert " " not in str(exe) and " " not in str(me), "路径里有空格，WMI 命令行需要另加引号"
    cmd = f"(Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{{CommandLine='{exe} {me} chain'; CurrentDirectory='{W}'}}).ProcessId"
    print("chain pid", run(["powershell", "-NoProfile", "-Command", cmd], timeout=60))
    time.sleep(20)
    print((W / "status" / "events.log").read_text(encoding="utf-8")[-600:] if (W / "status" / "events.log").exists() else "events.log 还没出现")


if __name__ == "__main__":
    for d in ("status", "logs", "mon", "out"):
        (W / d).mkdir(parents=True, exist_ok=True)
    act = sys.argv[1] if len(sys.argv) > 1 else "status"
    if act == "status":
        p = W / "status" / "latest.txt"
        print(p.read_text(encoding="utf-8") if p.exists() else "还没有状态文件")
    else:
        dict(smoke=smoke, launch=launch, chain=chain)[act]()
