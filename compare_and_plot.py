import pandas as pd, matplotlib.pyplot as plt, os, math

seq_cpu_file = "results_cpu_sequential.csv"
par_cpu_file = "results_cpu.csv"
gpu_file = "results_gpu.csv"

if not os.path.exists(seq_cpu_file) or not os.path.exists(par_cpu_file) or not os.path.exists(gpu_file):
    print("Run cpu_sequential_runner.py, cpu_pbkdf2_runner.py, and gpu_pbkdf2_runner.py first.")
    raise SystemExit

seq_cpu = pd.read_csv(seq_cpu_file)
par_cpu = pd.read_csv(par_cpu_file)
gpu = pd.read_csv(gpu_file)

c = seq_cpu.iloc[-1]
p = par_cpu.iloc[-1]
g = gpu.iloc[-1]

t_seq = float(c['elapsed_total_s'])
t_par = float(p['elapsed_total_s'])
t_gpu = float(g['elapsed_total_s'])
n_candidates = int(c['total_candidates'])

cpu_workers = 8
gpu_threads = 256  # approximate

speedup_cpu = t_seq / t_par if t_par > 0 else float('inf')
efficiency_cpu = speedup_cpu / cpu_workers
speedup_gpu = t_seq / t_gpu if t_gpu > 0 else float('inf')
efficiency_gpu = speedup_gpu / gpu_threads  # rough estimate

print("Sequential CPU total time (s):", t_seq)
print("Parallel CPU total time (s):", t_par)
print("GPU total time (s):", t_gpu)
print("Total candidates processed:", n_candidates)
print("CPU Speedup (T_seq / T_par):", speedup_cpu)
print("CPU Efficiency (speedup / workers):", efficiency_cpu)
print("GPU Speedup (T_seq / T_gpu):", speedup_gpu)
print("GPU Efficiency (speedup / threads):", efficiency_gpu)

plt.figure(figsize=(8,4))
plt.bar(["Sequential CPU", "Parallel CPU", "GPU"], [t_seq, t_par, t_gpu])
plt.ylabel("Time (s)")
plt.title("Total time: Sequential CPU vs Parallel CPU vs GPU")
plt.savefig("time_bar.png")

plt.figure(figsize=(8,4))
plt.bar(["CPU Speedup", "GPU Speedup"], [speedup_cpu, speedup_gpu])
plt.ylabel("Speedup")
plt.title("Speedup Comparison")
plt.savefig("speedup_bar.png")

print("Saved time_bar.png and speedup_bar.png")
