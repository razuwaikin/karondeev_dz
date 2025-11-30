#!/usr/bin/env python3
import pandas as pd, matplotlib.pyplot as plt, os, math

cpu_file = "results_cpu.csv"
gpu_file = "results_gpu.csv"

if not os.path.exists(cpu_file) or not os.path.exists(gpu_file):
    print("Run both cpu_pbkdf2_runner.py and gpu_pbkdf2_runner.py first.")
    raise SystemExit

cpu = pd.read_csv(cpu_file)
gpu = pd.read_csv(gpu_file)

# take last rows (latest runs)
c = cpu.iloc[-1]
g = gpu.iloc[-1]

t_cpu = float(c['elapsed_total_s'])
t_gpu = float(g['elapsed_total_s'])
n_candidates = int(c['total_candidates'])

# Workers:
# CPU single-thread 'workers' = 1
cpu_workers = 1
# GPU effective workers = number of active threads (approx): blocks*threads
batch_size = int(g['batch_size'])
# estimate threads used (we know batch_size); treat GPU workers = batch_size
gpu_workers = batch_size

speedup = t_cpu / t_gpu if t_gpu>0 else float('inf')
efficiency = speedup / gpu_workers

print("CPU total time (s):", t_cpu)
print("GPU total time (s):", t_gpu)
print("Total candidates processed:", n_candidates)
print("Speedup (T_cpu / T_gpu):", speedup)
print("Efficiency (speedup / workers):", efficiency)

# simple bar chart
plt.figure(figsize=(6,4))
plt.bar(["CPU","GPU"], [t_cpu, t_gpu])
plt.ylabel("Time (s)")
plt.title("Total time: CPU vs GPU")
plt.savefig("time_bar.png")

# speedup plot (single point)
plt.figure(figsize=(6,4))
plt.bar(["speedup"], [speedup])
plt.ylabel("Speedup")
plt.savefig("speedup_bar.png")

print("Saved time_bar.png and speedup_bar.png")
