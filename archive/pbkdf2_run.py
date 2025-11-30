#!/usr/bin/env python3
"""
pbkdf2_run.py

Тест производительности PBKDF2-HMAC-SHA256:
 - один фиксированный пароль (~15 символов)
 - много повторов N (симулируем нагрузку)
 - последовательный прогон
 - параллельный прогон (CPU)
 - результаты сохраняются в CSV

НЕ выполняет подбор паролей.
"""

import time
import hashlib
import csv
import secrets
from concurrent.futures import ProcessPoolExecutor

PASSWORD = b"my_safe_test_pwd"   # примерно 15 символов
DKLEN = 32


def pbkdf2_once(salt, iterations):
    return hashlib.pbkdf2_hmac("sha256", PASSWORD, salt, iterations, DKLEN)


def run_sequential(n_tasks, iterations):
    t0 = time.perf_counter()
    for _ in range(n_tasks):
        salt = secrets.token_bytes(16)
        pbkdf2_once(salt, iterations)
    t1 = time.perf_counter()
    return t1 - t0


def run_parallel(n_tasks, iterations, workers):
    salts = [secrets.token_bytes(16) for _ in range(n_tasks)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda s: pbkdf2_once(s, iterations), salts))
    t1 = time.perf_counter()
    return t1 - t0


def save_csv(filename, mode, n_tasks, iterations, workers, elapsed):
    with open(filename, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([mode, n_tasks, iterations, workers, elapsed])


def main():
    n_tasks_list = [200, 400, 800]      # можно менять
    iterations_list = [5000, 10000]     # PBKDF2 cost
    workers_list = [2, 4, 8]            # кол-во процессов

    for n_tasks in n_tasks_list:
        for iters in iterations_list:

            # последовательный
            t_seq = run_sequential(n_tasks, iters)
            save_csv("results_seq.csv", "seq", n_tasks, iters, 1, t_seq)
            print(f"[SEQ] tasks={n_tasks} iter={iters}: {t_seq:.3f} s")

            # параллельный
            for w in workers_list:
                t_par = run_parallel(n_tasks, iters, w)
                save_csv("results_par.csv", "par", n_tasks, iters, w, t_par)
                print(f"[PAR] tasks={n_tasks} iter={iters} workers={w}: {t_par:.3f} s")


if __name__ == "__main__":
    main()
