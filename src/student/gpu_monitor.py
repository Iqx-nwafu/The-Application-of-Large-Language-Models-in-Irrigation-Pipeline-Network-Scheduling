"""训练期间 GPU 监控：nvidia-smi（0.5 s）与 Windows 显存计数器（1 s，含共享内存溢出）逐行 fsync 落盘，蓝屏后可查崩前状态。
用法：python src/student/gpu_monitor.py <输出前缀>   → <前缀>_smi.csv、<前缀>_wddm.csv；Ctrl+C 或杀进程结束。"""
import os
import subprocess
import sys
import threading

out = sys.argv[1]
SMI = ["nvidia-smi", "--query-gpu=timestamp,power.draw,temperature.gpu,clocks.sm,clocks.mem,utilization.gpu,memory.used,pstate,clocks_throttle_reasons.active",
       "--format=csv,noheader", "-lms", "500"]
WDDM = ["typeperf", r"\GPU Adapter Memory(*)\Dedicated Usage", r"\GPU Adapter Memory(*)\Shared Usage", "-si", "1"]


def pump(cmd, path):
    with open(path, "a", encoding="utf-8") as f, subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, errors="replace",
                                                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as p:
        for line in p.stdout:
            f.write(line); f.flush(); os.fsync(f.fileno())


ts = [threading.Thread(target=pump, args=(c, f"{out}_{k}.csv"), daemon=True) for c, k in ((SMI, "smi"), (WDDM, "wddm"))]
for t in ts:
    t.start()
for t in ts:
    t.join()
