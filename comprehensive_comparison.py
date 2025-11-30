import pandas as pd
import matplotlib.pyplot as plt
import os

folders = ["full_search_results", "early_termination_results", "typical_results"]
labels = ["Full Search", "Early Termination", "Typical"]

results = []
for folder, label in zip(folders, labels):
    seq_file = os.path.join(folder, "results_cpu_sequential.csv")
    par_file = os.path.join(folder, "results_cpu.csv")

    if os.path.exists(seq_file) and os.path.exists(par_file):
        seq_df = pd.read_csv(seq_file)
        par_df = pd.read_csv(par_file)

        seq_time = seq_df.iloc[-1]['elapsed_total_s']
        par_time = par_df.iloc[-1]['elapsed_total_s']
        seq_candidates = seq_df.iloc[-1]['total_candidates']
        par_candidates = par_df.iloc[-1]['total_candidates']

        seq_speed = seq_candidates / seq_time if seq_time > 0 else 0
        par_speed = par_candidates / par_time if par_time > 0 else 0

        acceleration = seq_time / par_time if par_time > 0 else 0
        efficiency = acceleration / 8

        results.append({
            'label': label,
            'seq_time': seq_time,
            'par_time': par_time,
            'seq_speed': seq_speed,
            'par_speed': par_speed,
            'acceleration': acceleration,
            'efficiency': efficiency
        })

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Comprehensive PBKDF2 Benchmark Comparison\nSequential vs Parallel CPU (8 processes)', fontsize=16)

labels_plot = [r['label'] for r in results]
seq_times = [r['seq_time'] for r in results]
par_times = [r['par_time'] for r in results]

x = range(len(labels_plot))
ax1.bar([i - 0.2 for i in x], seq_times, 0.4, label='Sequential CPU', color='blue', alpha=0.7)
ax1.bar([i + 0.2 for i in x], par_times, 0.4, label='Parallel CPU', color='red', alpha=0.7)
ax1.set_xticks(x)
ax1.set_xticklabels(labels_plot)
ax1.set_ylabel('Time (seconds)')
ax1.set_title('Execution Time Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

seq_speeds = [r['seq_speed'] for r in results]
par_speeds = [r['par_speed'] for r in results]

ax2.bar([i - 0.2 for i in x], seq_speeds, 0.4, label='Sequential CPU', color='blue', alpha=0.7)
ax2.bar([i + 0.2 for i in x], par_speeds, 0.4, label='Parallel CPU', color='red', alpha=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels(labels_plot)
ax2.set_ylabel('Speed (candidates/second)')
ax2.set_title('Processing Speed Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)

accelerations = [r['acceleration'] for r in results]
bars = ax3.bar(labels_plot, accelerations, color='green', alpha=0.7)
ax3.set_ylabel('Acceleration (t_sequential / t_parallel)')
ax3.set_title('Acceleration Metric')
ax3.grid(True, alpha=0.3)

for bar, acc in zip(bars, accelerations):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{acc:.2f}x', ha='center', va='bottom')

efficiencies = [r['efficiency'] for r in results]
bars = ax4.bar(labels_plot, efficiencies, color='orange', alpha=0.7)
ax4.set_ylabel('Efficiency (acceleration / 8 processes)')
ax4.set_title('Efficiency Metric')
ax4.grid(True, alpha=0.3)

for bar, eff in zip(bars, efficiencies):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{eff:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('comprehensive_comparison.png', dpi=300, bbox_inches='tight')
print("Saved comprehensive_comparison.png")

print("\n" + "="*80)
print("COMPREHENSIVE BENCHMARK SUMMARY")
print("="*80)
print(f"{'Scenario':<20} {'Seq Time':<10} {'Par Time':<10} {'Seq Speed':<12} {'Par Speed':<12} {'Accel':<8} {'Eff':<8}")
print("-"*80)
for r in results:
    print(f"{r['label']:<20} {r['seq_time']:<10.2f} {r['par_time']:<10.2f} {r['seq_speed']:<12.0f} {r['par_speed']:<12.0f} {r['acceleration']:<8.2f} {r['efficiency']:<8.3f}")
print("="*80)

avg_seq_time = sum(r['seq_time'] for r in results) / len(results)
avg_par_time = sum(r['par_time'] for r in results) / len(results)
avg_seq_speed = sum(r['seq_speed'] for r in results) / len(results)
avg_par_speed = sum(r['par_speed'] for r in results) / len(results)
avg_acceleration = sum(r['acceleration'] for r in results) / len(results)
avg_efficiency = sum(r['efficiency'] for r in results) / len(results)

print("\nOVERALL AVERAGES:")
print(f"Average Sequential Time: {avg_seq_time:.2f}s")
print(f"Average Parallel Time: {avg_par_time:.2f}s")
print(f"Average Sequential Speed: {avg_seq_speed:.0f} cand/s")
print(f"Average Parallel Speed: {avg_par_speed:.0f} cand/s")
print(f"Average Acceleration: {avg_acceleration:.2f}x")
print(f"Average Efficiency: {avg_efficiency:.3f}")
print("="*80)