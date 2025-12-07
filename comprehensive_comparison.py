import pandas as pd
import matplotlib.pyplot as plt
import os

folders = ["full_search_results", "early_termination_results", "typical_results"]
labels = ["Full Search", "Early Termination", "Typical"]

results = []
for folder, label in zip(folders, labels):
    seq_file = os.path.join(folder, "results_cpu_sequential.csv")
    par_file = os.path.join(folder, "results_cpu.csv")
    gpu_file = os.path.join(folder, "results_gpu.csv")

    if os.path.exists(seq_file) and os.path.exists(par_file) and os.path.exists(gpu_file):
        seq_df = pd.read_csv(seq_file)
        par_df = pd.read_csv(par_file)
        gpu_df = pd.read_csv(gpu_file)

        seq_time = seq_df.iloc[-1]['elapsed_total_s']
        par_time = par_df.iloc[-1]['elapsed_total_s']
        gpu_time = gpu_df.iloc[-1]['elapsed_total_s']
        seq_candidates = seq_df.iloc[-1]['total_candidates']
        par_candidates = par_df.iloc[-1]['total_candidates']
        gpu_candidates = gpu_df.iloc[-1]['total_candidates']

        seq_speed = seq_candidates / seq_time if seq_time > 0 else 0
        par_speed = par_candidates / par_time if par_time > 0 else 0
        gpu_speed = gpu_candidates / gpu_time if gpu_time > 0 else 0

        acceleration_cpu = seq_time / par_time if par_time > 0 else 0
        efficiency_cpu = acceleration_cpu / 8
        acceleration_gpu = seq_time / gpu_time if gpu_time > 0 else 0
        efficiency_gpu = acceleration_gpu / 256  # approximate threads per block

        results.append({
            'label': label,
            'seq_time': seq_time,
            'par_time': par_time,
            'gpu_time': gpu_time,
            'seq_speed': seq_speed,
            'par_speed': par_speed,
            'gpu_speed': gpu_speed,
            'acceleration_cpu': acceleration_cpu,
            'efficiency_cpu': efficiency_cpu,
            'acceleration_gpu': acceleration_gpu,
            'efficiency_gpu': efficiency_gpu
        })

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Comprehensive PBKDF2 Benchmark Comparison\nSequential CPU vs Parallel CPU (8 processes) vs GPU', fontsize=16)

labels_plot = [r['label'] for r in results]
seq_times = [r['seq_time'] for r in results]
par_times = [r['par_time'] for r in results]
gpu_times = [r['gpu_time'] for r in results]

x = range(len(labels_plot))
width = 0.25
ax1.bar([i - width for i in x], seq_times, width, label='Sequential CPU', color='blue', alpha=0.7)
ax1.bar([i for i in x], par_times, width, label='Parallel CPU', color='red', alpha=0.7)
ax1.bar([i + width for i in x], gpu_times, width, label='GPU', color='green', alpha=0.7)
ax1.set_xticks(x)
ax1.set_xticklabels(labels_plot)
ax1.set_ylabel('Time (seconds)')
ax1.set_title('Execution Time Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

seq_speeds = [r['seq_speed'] for r in results]
par_speeds = [r['par_speed'] for r in results]
gpu_speeds = [r['gpu_speed'] for r in results]

ax2.bar([i - width for i in x], seq_speeds, width, label='Sequential CPU', color='blue', alpha=0.7)
ax2.bar([i for i in x], par_speeds, width, label='Parallel CPU', color='red', alpha=0.7)
ax2.bar([i + width for i in x], gpu_speeds, width, label='GPU', color='green', alpha=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels(labels_plot)
ax2.set_ylabel('Speed (candidates/second)')
ax2.set_title('Processing Speed Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)

accelerations_cpu = [r['acceleration_cpu'] for r in results]
accelerations_gpu = [r['acceleration_gpu'] for r in results]

x_acc = range(len(labels_plot))
width_acc = 0.35
bars_cpu = ax3.bar([i - width_acc/2 for i in x_acc], accelerations_cpu, width_acc, label='CPU Acceleration', color='green', alpha=0.7)
bars_gpu = ax3.bar([i + width_acc/2 for i in x_acc], accelerations_gpu, width_acc, label='GPU Acceleration', color='purple', alpha=0.7)
ax3.set_xticks(x_acc)
ax3.set_xticklabels(labels_plot)
ax3.set_ylabel('Acceleration (t_sequential / t_parallel)')
ax3.set_title('Acceleration Metric')
ax3.legend()
ax3.grid(True, alpha=0.3)

for bar, acc in zip(bars_cpu, accelerations_cpu):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{acc:.2f}x', ha='center', va='bottom')
for bar, acc in zip(bars_gpu, accelerations_gpu):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{acc:.2f}x', ha='center', va='bottom')

efficiencies_cpu = [r['efficiency_cpu'] for r in results]
efficiencies_gpu = [r['efficiency_gpu'] for r in results]

bars_cpu_eff = ax4.bar([i - width_acc/2 for i in x_acc], efficiencies_cpu, width_acc, label='CPU Efficiency', color='orange', alpha=0.7)
bars_gpu_eff = ax4.bar([i + width_acc/2 for i in x_acc], efficiencies_gpu, width_acc, label='GPU Efficiency', color='brown', alpha=0.7)
ax4.set_xticks(x_acc)
ax4.set_xticklabels(labels_plot)
ax4.set_ylabel('Efficiency')
ax4.set_title('Efficiency Metric')
ax4.legend()
ax4.grid(True, alpha=0.3)

for bar, eff in zip(bars_cpu_eff, efficiencies_cpu):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{eff:.3f}', ha='center', va='bottom')
for bar, eff in zip(bars_gpu_eff, efficiencies_gpu):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{eff:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('comprehensive_comparison.png', dpi=300, bbox_inches='tight')
print("Saved comprehensive_comparison.png")

print("\n" + "="*100)
print("COMPREHENSIVE BENCHMARK SUMMARY")
print("="*100)
print(f"{'Scenario':<20} {'Seq Time':<10} {'Par Time':<10} {'GPU Time':<10} {'Seq Speed':<12} {'Par Speed':<12} {'GPU Speed':<12} {'CPU Accel':<10} {'GPU Accel':<10} {'CPU Eff':<8} {'GPU Eff':<8}")
print("-"*100)
for r in results:
    print(f"{r['label']:<20} {r['seq_time']:<10.2f} {r['par_time']:<10.2f} {r['gpu_time']:<10.2f} {r['seq_speed']:<12.0f} {r['par_speed']:<12.0f} {r['gpu_speed']:<12.0f} {r['acceleration_cpu']:<10.2f} {r['acceleration_gpu']:<10.2f} {r['efficiency_cpu']:<8.3f} {r['efficiency_gpu']:<8.3f}")
print("="*100)

avg_seq_time = sum(r['seq_time'] for r in results) / len(results)
avg_par_time = sum(r['par_time'] for r in results) / len(results)
avg_gpu_time = sum(r['gpu_time'] for r in results) / len(results)
avg_seq_speed = sum(r['seq_speed'] for r in results) / len(results)
avg_par_speed = sum(r['par_speed'] for r in results) / len(results)
avg_gpu_speed = sum(r['gpu_speed'] for r in results) / len(results)
avg_acceleration_cpu = sum(r['acceleration_cpu'] for r in results) / len(results)
avg_acceleration_gpu = sum(r['acceleration_gpu'] for r in results) / len(results)
avg_efficiency_cpu = sum(r['efficiency_cpu'] for r in results) / len(results)
avg_efficiency_gpu = sum(r['efficiency_gpu'] for r in results) / len(results)

print("\nOVERALL AVERAGES:")
print(f"Average Sequential Time: {avg_seq_time:.2f}s")
print(f"Average Parallel Time: {avg_par_time:.2f}s")
print(f"Average GPU Time: {avg_gpu_time:.2f}s")
print(f"Average Sequential Speed: {avg_seq_speed:.0f} cand/s")
print(f"Average Parallel Speed: {avg_par_speed:.0f} cand/s")
print(f"Average GPU Speed: {avg_gpu_speed:.0f} cand/s")
print(f"Average CPU Acceleration: {avg_acceleration_cpu:.2f}x")
print(f"Average GPU Acceleration: {avg_acceleration_gpu:.2f}x")
print(f"Average CPU Efficiency: {avg_efficiency_cpu:.3f}")
print(f"Average GPU Efficiency: {avg_efficiency_gpu:.3f}")
print("="*100)