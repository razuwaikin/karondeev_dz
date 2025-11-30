#!/usr/bin/env python3
# cpu_pbkdf2_runner.py
import json, os, time, csv, secrets, sys
import hashlib
import numpy as np

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

import string
CHARS = string.printable[:-6]

# salt
salt = bytes.fromhex(cfg.get("salt_hex")) if cfg.get("salt_hex") else secrets.token_bytes(12)

# target
if TARGET:
    target_password = TARGET
else:
    # if not provided, replicate same generation method: read GPU file if exists
    # but for simplicity, generate random here (should be same as GPU run if you want to compare: set target_password in config)
    target_password = ''.join(secrets.choice(CHARS) for _ in range(L))
    print("CPU: generated target:", target_password)
target_hash = hashlib.pbkdf2_hmac("sha256", target_password.encode(), salt, ITER, DKLEN)
print("Target hash hex:", target_hash.hex())

def gen_candidate(length):
    return ''.join(secrets.choice(CHARS) for _ in range(length))

def run():
    found = False
    total = 0
    timings = []
    found_info = None

    for batch_idx in range(N_BATCHES):
        # generate batch list of strings
        batch = [gen_candidate(L) for _ in range(BATCH_SIZE)]
        if INSERT_TARGET and batch_idx == secrets.randbelow(N_BATCHES):
            pos = secrets.randbelow(BATCH_SIZE)
            batch[pos] = target_password
            print(f"Inserted target at CPU batch {batch_idx} pos {pos}")

        t0 = time.perf_counter()
        for i, p in enumerate(batch):
            h = hashlib.pbkdf2_hmac('sha256', p.encode(), salt, ITER, DKLEN)
            total += 1
            if h == target_hash:
                found = True
                found_info = (batch_idx, i)
                break
        t1 = time.perf_counter()
        timings.append(t1 - t0)
        print(f"[CPU] batch {batch_idx} done: time={(t1-t0):.4f}s (cumulative {total})")
        if found:
            print("FOUND on CPU:", found_info)
            break

    avg = sum(timings)/len(timings) if timings else None
    exists = os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp","n_batches","batch_size","iterations","dklen","total_candidates","found","found_batch","found_pos","elapsed_avg_s","elapsed_total_s","target_password","target_hash_hex"])
        w.writerow([time.time(), N_BATCHES, BATCH_SIZE, ITER, DKLEN, total, bool(found), found_info[0] if found else -1, found_info[1] if found else -1, avg, sum(timings), target_password, target_hash.hex()])
    print("CPU run complete. Saved to", OUT_CSV)

if __name__ == "__main__":
    run()
