import pandas as pd
import matplotlib.pyplot as plt

# Файлы с результатами
seq_file = "results_seq.csv"
par_file = "results_par.csv"

seq = pd.read_csv(seq_file, header=None,
                  names=["mode", "tasks", "iters", "workers", "time"])
par = pd.read_csv(par_file, header=None,
                  names=["mode", "tasks", "iters", "workers", "time"])

# Для удобства — объединим
all_data = pd.concat([seq, par])

# ГРАФИК 1 — время vs число воркеров
for tasks in all_data["tasks"].unique():
    for it in all_data["iters"].unique():
        df = all_data[(all_data.tasks == tasks) & (all_data.iters == it)]
        plt.figure()
        seq_t = df[df.mode == "seq"].time.values[0]
        df_par = df[df.mode == "par"].sort_values("workers")

        plt.plot([1], [seq_t], 'o', label="seq")
        plt.plot(df_par["workers"], df_par["time"], '-o', label="parallel")

        plt.title(f"Time vs Workers (tasks={tasks}, iter={it})")
        plt.xlabel("Workers")
        plt.ylabel("Time (s)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"plot_time_tasks{tasks}_iter{it}.png")

# ГРАФИК 2 — ускорение (speedup = seq / par)
for tasks in all_data["tasks"].unique():
    for it in all_data["iters"].unique():
        df = all_data[(all_data.tasks == tasks) & (all_data.iters == it)]
        seq_t = df[df.mode == "seq"].time.values[0]
        df_par = df[df.mode == "par"].sort_values("workers")

        speedup = seq_t / df_par["time"]
        plt.figure()
        plt.plot(df_par["workers"], speedup, '-o')
        plt.title(f"Speedup (tasks={tasks}, iter={it})")
        plt.xlabel("Workers")
        plt.ylabel("Speedup")
        plt.grid(True)
        plt.savefig(f"plot_speedup_tasks{tasks}_iter{it}.png")
