import json, os, time, csv, sys, math, argparse
import numpy as np
import secrets
import hashlib
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config.json', help='Path to config file')
parser.add_argument('--output', default='results_gpu.csv', help='Path to output CSV')
args = parser.parse_args()

CFG_FNAME = args.config
OUT_CSV = args.output

with open(CFG_FNAME, "r") as f:
    cfg = json.load(f)

BATCH_SIZE = int(cfg.get("batch_size", 8192))
N_BATCHES = int(cfg.get("n_batches", 16))
L = int(cfg.get("length", 10))
ITER = int(cfg.get("iterations", 5000))
DKLEN = int(cfg.get("dklen", 32))
INSERT_TARGET = bool(cfg.get("insert_target_in_batch", True))
GPU_WORKERS = min(int(cfg.get("gpu_threads_per_block", 256)), 16)

salt = bytes.fromhex(cfg.get("salt_hex")) if cfg.get("salt_hex") else secrets.token_bytes(12)

if cfg.get("target_password"):
    target_password = cfg.get("target_password")
else:
    import string
    CHARS = string.printable[:-6]
    target_password = ''.join(secrets.choice(CHARS) for _ in range(L))
    print("GPU Simulator: generated target:", target_password)

target_hash = hashlib.pbkdf2_hmac("sha256", target_password.encode(), salt, ITER, DKLEN)
print("Target hash (hex):", target_hash.hex())

import string
CHARS = string.printable[:-6]

def gen_candidate(length):
    return ''.join(secrets.choice(CHARS) for _ in range(length))

def compute_hash(p):
    return hashlib.pbkdf2_hmac('sha256', p.encode(), salt, ITER, DKLEN)

def gen_batch(batch_size, length):
    return [gen_candidate(length) for _ in range(batch_size)]

def insert_target_into_batch(batch, target_password):
    pos = secrets.randbelow(len(batch))
    batch[pos] = target_password
    return pos

def run():
    found = False
    total_candidate_count = 0
    timings = []
    found_info = None

    with multiprocessing.Pool(processes=GPU_WORKERS) as pool:
        for batch_index in range(N_BATCHES):
            batch = gen_batch(BATCH_SIZE, L)
            if INSERT_TARGET and batch_index == secrets.randbelow(N_BATCHES):
                pos = insert_target_into_batch(batch, target_password)
                print(f"Inserted target into batch {batch_index} at pos {pos}")

            t0 = time.perf_counter()
            hashes = pool.map(compute_hash, batch)
            t1 = time.perf_counter()
            elapsed_s = t1 - t0
            timings.append(elapsed_s)

            for i, h in enumerate(hashes):
                if h == target_hash:
                    found = True
                    found_info = (batch_index, i)
                    break

            total_candidate_count += BATCH_SIZE
            print(f"[GPU Sim] batch {batch_index} done: time={elapsed_s:.4f}s (cumulative candidates={total_candidate_count})")
            if found:
                print("FOUND on GPU Sim in batch", found_info)
                break

    avg_time = sum(timings)/len(timings) if timings else None
    header = ["timestamp","n_batches","batch_size","iterations","dklen","total_candidates","found","found_batch","found_pos","elapsed_avg_s","elapsed_total_s","target_password","target_hash_hex"]
    exists = os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow([time.time(), N_BATCHES, BATCH_SIZE, ITER, DKLEN, total_candidate_count, bool(found),
                    found_info[0] if found else -1, found_info[1] if found else -1, avg_time, sum(timings),
                    target_password, target_hash.hex()])
    print("GPU simulation run complete. Saved to", OUT_CSV)

if __name__ == "__main__":
    run()