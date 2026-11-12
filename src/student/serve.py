"""起 / 停本地 llama-server（Windows 原生 b12032，WMI 起进程，脱离会话），评测与部署测速共用同一组参数，参数写进日志便于复现。
  python src/student/serve.py start <gguf> [--port 8091] [--ctx 32768] [--kv f16|q8_0] [--ngl 99] [--threads 4]
  python src/student/serve.py stop [--port 8091]
固定：-np 1（单槽，正式时延口径）、-fa on（11-02 实测快约 40%）、--jinja、不强加思考块（请求带 enable_thinking=true，见 eval_student 说明）、
seed 0。--ngl 0 即纯 CPU。日志 <LLAMA_LOG_DIR>/server_<port>.log；起来后等 /health 就绪再返回。"""
import argparse
import json
import os
import subprocess
import time
import urllib.request

BIN = os.environ.get("LLAMA_SERVER", "llama-server.exe")            # llama.cpp b12032


def ps(cmd):
    return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True).stdout.strip()


def pids(port):
    out = ps(f"Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" | Where-Object {{ $_.CommandLine -match '--port {port}( |$)' }} "
             f"| ForEach-Object {{ $_.ProcessId }}")
    return [int(x) for x in out.split()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["start", "stop"])
    ap.add_argument("gguf", nargs="?")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--kv", default="f16", help="KV 缓存类型（-ctk/-ctv）：bf16 权重 + 32k 上下文 + f16 KV 约 7.4 GB，8 GB 放不下时用 q8_0")
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--threads", type=int, default=4, help="全部层在 GPU 时 CPU 线程影响很小；默认 24 线程会和 GA 抢 CPU（已知坑 20）")
    a = ap.parse_args()
    if a.action == "stop":
        for p in pids(a.port):
            ps(f"Stop-Process -Id {p} -Force")
        print("已停", a.port)
        return
    if pids(a.port):
        raise SystemExit(f"端口 {a.port} 已有 llama-server，先 stop")
    log = os.path.join(os.environ.get("LLAMA_LOG_DIR", os.getcwd()), f"server_{a.port}.log")
    args = [BIN, "-m", a.gguf, "--port", str(a.port), "-c", str(a.ctx), "-np", "1", "-ngl", str(a.ngl), "-fa", "on", "--jinja", "-ctk", a.kv, "-ctv", a.kv, "-t", str(a.threads), "--seed", "0", "--log-file", log]
    cl = " ".join(f'"{x}"' if " " in x else x for x in args)
    r = ps(f"(Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{{CommandLine='{cl}'; CurrentDirectory='{os.getcwd()}'}}).ProcessId")
    print(f"pid {r}：{cl}", flush=True)
    for _ in range(180):
        try:
            if json.load(urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2)).get("status") == "ok":
                print("就绪", flush=True)
                return
        except OSError:
            pass
        time.sleep(1)
    raise SystemExit(f"3 分钟内未就绪，看 {log}")


if __name__ == "__main__":
    main()
