#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

seq = pd.read_csv("results_seq.csv")
par = pd.read_csv("results_par.csv")
all_df = pd.concat([seq, par], ignore_index=True)

for n in all_df['n_tasks'].unique():
    for it in all_df['iterations'].unique():
        sub = all_df[(all_df['n_tasks']==n) & (all_df['iterations']==it)]
        seq_row = sub[sub['mode']=='seq']
        if seq_row.empty:
            continue
        seq_time = float(seq_row.iloc[0]['elapsed_s'])
        par_rows = sub[sub['mode']=='par'].sort_values('workers')
        if par_rows.empty:
            continue
        workers = par_rows['workers'].astype(int).tolist()
        par_times = par_rows['elapsed_s'].astype(float).tolist()
        # time plot
        plt.figure()
        plt.plot([1] + workers, [seq_time] + par_times, marker='o')
        plt.xlabel('workers')
        plt.ylabel('time (s)')
        plt.title(f"time vs workers (tasks={n}, iters={it})")
        plt.grid(True)
        plt.savefig(f"time_tasks{n}_iters{it}.png")

        # speedup plot
        speedups = [seq_time / t for t in par_times]
        plt.figure()
        plt.plot(workers, speedups, marker='o')
        plt.xlabel('workers')
        plt.ylabel('speedup')
        plt.title(f"speedup (tasks={n}, iters={it})")
        plt.grid(True)
        plt.savefig(f"speedup_tasks{n}_iters{it}.png")
