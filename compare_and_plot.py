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

# Get new metrics
seq_speed = c.get('passwords_per_second', n_candidates / t_seq if t_seq > 0 else 0)
par_speed = p.get('passwords_per_second', n_candidates / t_par if t_par > 0 else 0)
gpu_speed = g.get('passwords_per_second', n_candidates / t_gpu if t_gpu > 0 else 0)

seq_time_per_pass = c.get('time_per_password', t_seq / n_candidates if n_candidates > 0 else 0)
par_time_per_pass = p.get('time_per_password', t_par / n_candidates if n_candidates > 0 else 0)
gpu_time_per_pass = g.get('time_per_password', t_gpu / n_candidates if n_candidates > 0 else 0)

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
print("Sequential CPU speed (pass/sec):", seq_speed)
print("Parallel CPU speed (pass/sec):", par_speed)
print("GPU speed (pass/sec):", gpu_speed)
print("Sequential CPU time per password (s):", seq_time_per_pass)
print("Parallel CPU time per password (s):", par_time_per_pass)
print("GPU time per password (s):", gpu_time_per_pass)
print("CPU Speedup (T_seq / T_par):", speedup_cpu)
print("CPU Efficiency (speedup / workers):", efficiency_cpu)
print("GPU Speedup (T_seq / T_gpu):", speedup_gpu)
print("GPU Efficiency (speedup / threads):", efficiency_gpu)

# Hashcat comparison (typical values for PBKDF2-SHA256 with 5000 iterations)
hashcat_speed = 2000000  # ~2M hashes/sec on modern GPU
hashcat_time_per_pass = 1.0 / hashcat_speed
hashcat_speedup_vs_seq = seq_speed / hashcat_speed if hashcat_speed > 0 else 0
hashcat_speedup_vs_gpu = gpu_speed / hashcat_speed if hashcat_speed > 0 else 0

print("\nHashcat Comparison (estimated for similar PBKDF2-SHA256 setup):")
print("Hashcat speed (pass/sec):", hashcat_speed)
print("Hashcat time per password (s):", hashcat_time_per_pass)
print("Hashcat speedup vs Sequential CPU:", hashcat_speedup_vs_seq)
print("Hashcat speedup vs our GPU:", hashcat_speedup_vs_gpu)

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

plt.figure(figsize=(8,4))
bars = plt.bar(["Sequential CPU", "Parallel CPU", "GPU", "Hashcat (est.)"], [seq_speed, par_speed, gpu_speed, hashcat_speed])
plt.ylabel("Speed (passwords/second)")
plt.title("Processing Speed Comparison (including Hashcat estimate)")
plt.yscale('log')
# Add value labels on bars
for bar, speed in zip(bars, [seq_speed, par_speed, gpu_speed, hashcat_speed]):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{speed:,.0f}', ha='center', va='bottom')
plt.savefig("speed_comparison.png")

print("Saved time_bar.png, speedup_bar.png, and speed_comparison.png")
