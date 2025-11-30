import pandas as pd, matplotlib.pyplot as plt, os, math

cpu_file = "results_cpu_sequential.csv"
gpu_file = "results_cpu.csv"

if not os.path.exists(cpu_file) or not os.path.exists(gpu_file):
    print("Run both cpu_sequential_runner.py and cpu_pbkdf2_runner.py first.")
    raise SystemExit

cpu = pd.read_csv(cpu_file)
gpu = pd.read_csv(gpu_file)

c = cpu.iloc[-1]
g = gpu.iloc[-1]

t_cpu = float(c['elapsed_total_s'])
t_gpu = float(g['elapsed_total_s'])
n_candidates = int(c['total_candidates'])

cpu_workers = 1
gpu_workers = 8

speedup = t_cpu / t_gpu if t_gpu>0 else float('inf')
efficiency = speedup / gpu_workers

print("Sequential CPU total time (s):", t_cpu)
print("Parallel CPU total time (s):", t_gpu)
print("Total candidates processed:", n_candidates)
print("Speedup (T_sequential / T_parallel):", speedup)
print("Efficiency (speedup / workers):", efficiency)

plt.figure(figsize=(6,4))
plt.bar(["Sequential CPU","Parallel CPU"], [t_cpu, t_gpu])
plt.ylabel("Time (s)")
plt.title("Total time: Sequential CPU vs Parallel CPU")
plt.savefig("time_bar.png")

plt.figure(figsize=(6,4))
plt.bar(["speedup"], [speedup])
plt.ylabel("Speedup")
plt.savefig("speedup_bar.png")

print("Saved time_bar.png and speedup_bar.png")
