#!/usr/bin/env python3
"""
pbkdf2_benchmark_cpu.py
CPU benchmark: sequential vs parallel (multiprocessing).
Работать ТОЛЬКО локально.
"""
import time
import hashlib
import secrets
import csv
import os
from statistics import mean
from concurrent.futures import ProcessPoolExecutor

PASSWORD = b"local_test_pwd15"  # ~15 байт/символов
DKLEN = 32

def pbkdf2_once(args):
    password, salt, iterations, dklen = args
    return hashlib.pbkdf2_hmac("sha256", password, salt, iterations, dklen)

def run_sequential(n_tasks, iterations):
    salts = [secrets.token_bytes(16) for _ in range(n_tasks)]
    t0 = time.perf_counter()
    for s in salts:
        pbkdf2_once((PASSWORD, s, iterations, DKLEN))
    t1 = time.perf_counter()
    return t1 - t0

def run_parallel(n_tasks, iterations, workers):
    salts = [secrets.token_bytes(16) for _ in range(n_tasks)]
    args = [(PASSWORD, s, iterations, DKLEN) for s in salts]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(pbkdf2_once, args))
    t1 = time.perf_counter()
    return t1 - t0

def append_row(csv_fn, row):
    exists = os.path.exists(csv_fn)
    with open(csv_fn, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["mode","n_tasks","iterations","workers","elapsed_s","timestamp"])
        w.writerow(row)

def main():
    # параметры — настраивайте под машину
    n_tasks_list = [200, 400, 800]
    iterations_list = [5000, 10000]
    workers_list = [1, 2, 4, 8]  # 1→показываем последовательный результат как baseline

    for n_tasks in n_tasks_list:
        for it in iterations_list:
            # последовательный (workers=1)
            t_seq = run_sequential(n_tasks, it)
            print(f"[SEQ] tasks={n_tasks} iters={it} => {t_seq:.3f}s")
            append_row("results_seq.csv", ["seq", n_tasks, it, 1, f"{t_seq:.6f}", time.time()])

            # параллельный (CPU)
            for w in workers_list[1:]:
                t_par = run_parallel(n_tasks, it, w)
                print(f"[PAR] tasks={n_tasks} iters={it} workers={w} => {t_par:.3f}s")
                append_row("results_par.csv", ["par", n_tasks, it, w, f"{t_par:.6f}", time.time()])

if __name__ == "__main__":
    main()
