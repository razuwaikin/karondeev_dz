import json, os, time, csv, secrets, sys
import hashlib
import numpy as np
import multiprocessing

CFG_FNAME = "config.json"
OUT_CSV = "results_cpu.csv"

with open(CFG_FNAME, "r") as f:
    cfg = json.load(f)

BATCH_SIZE = int(cfg.get("batch_size", 8192))
N_BATCHES = int(cfg.get("n_batches", 16))
L = int(cfg.get("length", 10))
ITER = int(cfg.get("iterations", 5000))
DKLEN = int(cfg.get("dklen", 32))
TARGET = cfg.get("target_password", None)
INSERT_TARGET = bool(cfg.get("insert_target_in_batch", True))
CPU_WORKERS = int(cfg.get("cpu_workers", 8))

import string
CHARS = string.printable[:-6]

salt = bytes.fromhex(cfg.get("salt_hex")) if cfg.get("salt_hex") else secrets.token_bytes(12)

if TARGET:
    target_password = TARGET
else:
    target_password = ''.join(secrets.choice(CHARS) for _ in range(L))
    print("CPU: generated target:", target_password)
target_hash = hashlib.pbkdf2_hmac("sha256", target_password.encode(), salt, ITER, DKLEN)
print("Target hash hex:", target_hash.hex())

def gen_candidate(length):
    return ''.join(secrets.choice(CHARS) for _ in range(length))

def compute_hash(p):
    return hashlib.pbkdf2_hmac('sha256', p.encode(), salt, ITER, DKLEN)

def run():
    found = False
    total = 0
    timings = []
    found_info = None

    with multiprocessing.Pool(processes=CPU_WORKERS) as pool:
        for batch_idx in range(N_BATCHES):
            batch = [gen_candidate(L) for _ in range(BATCH_SIZE)]
            if INSERT_TARGET and batch_idx == secrets.randbelow(N_BATCHES):
                pos = secrets.randbelow(BATCH_SIZE)
                batch[pos] = target_password
                print(f"Inserted target at CPU batch {batch_idx} pos {pos}")

            t0 = time.perf_counter()
            hashes = pool.map(compute_hash, batch)
            t1 = time.perf_counter()
            timings.append(t1 - t0)

            for i, h in enumerate(hashes):
                total += 1
                if h == target_hash:
                    found = True
                    found_info = (batch_idx, i)
                    break

            print(f"[CPU] batch {batch_idx} done: time={(t1-t0):.4f}s (cumulative {total})")
            if found:
                print("FOUND on CPU:", found_info)
                break

    total_time = sum(timings)
    avg = total_time/len(timings) if timings else None
    time_per_password = total_time / total if total > 0 else None
    passwords_per_second = total / total_time if total_time > 0 else None
    # For CPU parallel, timing includes both computation and inter-process communication
    # For simplicity, we'll consider all time as operation time for now
    operation_time = total_time
    sending_time = 0.0

    exists = os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp","n_batches","batch_size","iterations","dklen","total_candidates","found","found_batch","found_pos","elapsed_avg_s","elapsed_total_s","time_per_password","passwords_per_second","operation_time_s","sending_time_s","target_password","target_hash_hex"])
        w.writerow([time.time(), N_BATCHES, BATCH_SIZE, ITER, DKLEN, total, bool(found), found_info[0] if found else -1, found_info[1] if found else -1, avg, total_time, time_per_password, passwords_per_second, operation_time, sending_time, target_password, target_hash.hex()])
    print("CPU run complete. Saved to", OUT_CSV)

if __name__ == "__main__":
    run()
