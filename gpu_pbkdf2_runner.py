import json, os, time, csv, sys, math
import numpy as np
import cupy as cp
import secrets
import hashlib

CFG_FNAME = "config.json"
OUT_CSV = "results_gpu.csv"

with open(CFG_FNAME, "r") as f:
    cfg = json.load(f)

BATCH_SIZE = int(cfg.get("batch_size", 8192))
N_BATCHES = int(cfg.get("n_batches", 16))
L = int(cfg.get("length", 10))
ITER = int(cfg.get("iterations", 5000))
DKLEN = int(cfg.get("dklen", 32))
TPB = int(cfg.get("gpu_threads_per_block", 256))
INSERT_TARGET = bool(cfg.get("insert_target_in_batch", True))

salt = bytes.fromhex(cfg.get("salt_hex")) if cfg.get("salt_hex") else secrets.token_bytes(12)

if cfg.get("target_password"):
    target_password = cfg.get("target_password")
else:
    import string
    CHARS = string.printable[:-6]
    target_password = ''.join(secrets.choice(CHARS) for _ in range(L))
    print("Generated target password:", target_password)

target_hash = hashlib.pbkdf2_hmac("sha256", target_password.encode(), salt, ITER, DKLEN)
print("Target hash (hex):", target_hash.hex())
cuda_source = r'''
#include <stdint.h>

// helpers
#define ROTR(x,n) (((x) >> (n)) | ((x) << (32-(n))))
#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define SIG0(x) (ROTR(x,2) ^ ROTR(x,13) ^ ROTR(x,22))
#define SIG1(x) (ROTR(x,6) ^ ROTR(x,11) ^ ROTR(x,25))
#define sig0(x) (ROTR(x,7) ^ ROTR(x,18) ^ ((x) >> 3))
#define sig1(x) (ROTR(x,17) ^ ROTR(x,19) ^ ((x) >> 10))

__device__ void sha256_transform(const uint8_t *chunk, uint32_t H[8]) {
    uint32_t w[64];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        w[i] = ((uint32_t)chunk[i*4] << 24) | ((uint32_t)chunk[i*4+1] << 16) | ((uint32_t)chunk[i*4+2] << 8) | ((uint32_t)chunk[i*4+3]);
    }
    for (int i = 16; i < 64; ++i) {
        w[i] = sig1(w[i-2]) + w[i-7] + sig0(w[i-15]) + w[i-16];
    }
    uint32_t a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
    const uint32_t K[64] = {
        0x428a2f98ul,0x71374491ul,0xb5c0fbcful,0xe9b5dba5ul,0x3956c25bul,0x59f111f1ul,0x923f82a4ul,0xab1c5ed5ul,
        0xd807aa98ul,0x12835b01ul,0x243185beul,0x550c7dc3ul,0x72be5d74ul,0x80deb1feul,0x9bdc06a7ul,0xc19bf174ul,
        0xe49b69c1ul,0xefbe4786ul,0x0fc19dc6ul,0x240ca1ccul,0x2de92c6ful,0x4a7484aal,0x5cb0a9dcul,0x76f988daul,
        0x983e5152ul,0xa831c66dul,0xb00327c8ul,0xbf597fc7ul,0xc6e00bf3ul,0xd5a79147ul,0x06ca6351ul,0x14292967ul,
        0x27b70a85ul,0x2e1b2138ul,0x4d2c6dfcul,0x53380d13ul,0x650a7354ul,0x766a0abbul,0x81c2c92eul,0x92722c85ul,
        0xa2bfe8a1ul,0xa81a664bul,0xc24b8b70ul,0xc76c51a3ul,0xd192e819ul,0xd6990624ul,0xf40e3585ul,0x106aa070ul,
        0x19a4c116ul,0x1e376c08ul,0x2748774cul,0x34b0bcb5ul,0x391c0cb3ul,0x4ed8aa4aul,0x5b9cca4ful,0x682e6ff3ul,
        0x748f82eeul,0x78a5636ful,0x84c87814ul,0x8cc70208ul,0x90befffamul,0xa4506cebul,0xbef9a3f7ul,0xc67178f2ul
    };
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        uint32_t T1 = h + SIG1(e) + CH(e,f,g) + K[i] + w[i];
        uint32_t T2 = SIG0(a) + MAJ(a,b,c);
        h = g; g = f; f = e; e = d + T1; d = c; c = b; b = a; a = T1 + T2;
    }
    H[0] += a; H[1] += b; H[2] += c; H[3] += d; H[4] += e; H[5] += f; H[6] += g; H[7] += h;
}

__device__ void sha256_single_block(const uint8_t *msg, int msglen, uint8_t out32[32]) {
    uint8_t block[64];
    int i;
    for (i = 0; i < msglen; ++i) block[i] = msg[i];
    block[i++] = 0x80;
    for (; i < 56; ++i) block[i] = 0;
    uint64_t bitlen = ((uint64_t)msglen) * 8;
    for (int j = 0; j < 8; ++j) block[63 - j] = (uint8_t)(bitlen >> (8*j));
    uint32_t H[8] = {
        0x6a09e667ul,0xbb67ae85ul,0x3c6ef372ul,0xa54ff53aul,
        0x510e527ful,0x9b05688cul,0x1f83d9abul,0x5be0cd19ul
    };
    sha256_transform(block, H);
    for (int k = 0; k < 8; ++k) {
        out32[k*4+0] = (uint8_t)(H[k] >> 24);
        out32[k*4+1] = (uint8_t)(H[k] >> 16);
        out32[k*4+2] = (uint8_t)(H[k] >> 8);
        out32[k*4+3] = (uint8_t)(H[k] >> 0);
    }
}

__device__ void hmac_sha256_small(const uint8_t *key, int key_len, const uint8_t *msg, int msg_len, uint8_t out32[32]) {
    uint8_t key_block[64];
    for (int i = 0; i < 64; ++i) key_block[i] = 0;
    for (int i = 0; i < key_len; ++i) key_block[i] = key[i];
    uint8_t ipad[64], opad[64];
    for (int i = 0; i < 64; ++i) {
        ipad[i] = key_block[i] ^ 0x36;
        opad[i] = key_block[i] ^ 0x5c;
    }
    // inner
    uint8_t inner[64 + 64];
    int inner_len = 0;
    for (int i = 0; i < 64; ++i) inner[inner_len++] = ipad[i];
    for (int i = 0; i < msg_len; ++i) inner[inner_len++] = msg[i];
    uint8_t inner_hash[32];
    sha256_single_block(inner, inner_len, inner_hash);
    // outer
    uint8_t outer[64 + 32];
    int outer_len = 0;
    for (int i = 0; i < 64; ++i) outer[outer_len++] = opad[i];
    for (int i = 0; i < 32; ++i) outer[outer_len++] = inner_hash[i];
    sha256_single_block(outer, outer_len, out32);
}

__device__ void xor32(uint8_t *dst, const uint8_t *src) {
    for (int i = 0; i < 32; ++i) dst[i] ^= src[i];
}

extern "C" __global__
void pbkdf2_kernel(const uint8_t *passwords, int pass_len,
                   const uint8_t *salts, int salt_len,
                   int iterations, uint8_t *out, int dklen, int n_tasks)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_tasks) return;

    const uint8_t *salt = salts + idx * salt_len;
    const uint8_t *pass = passwords + idx * pass_len;

    uint8_t T[32];
    uint8_t U[32];
    uint8_t buf[68]; // salt + 4

    for (int i = 0; i < salt_len; ++i) buf[i] = salt[i];
    buf[salt_len+0] = 0x00;
    buf[salt_len+1] = 0x00;
    buf[salt_len+2] = 0x00;
    buf[salt_len+3] = 0x01;
    int buf_len = salt_len + 4;

    hmac_sha256_small(pass, pass_len, buf, buf_len, U);
    for (int i = 0; i < 32; ++i) T[i] = U[i];

    for (int iter = 1; iter < iterations; ++iter) {
        hmac_sha256_small(pass, pass_len, U, 32, U);
        xor32(T, U);
    }

    uint8_t *out_ptr = out + idx * dklen;
    for (int i = 0; i < dklen; ++i) out_ptr[i] = T[i];
}
'''

print("Compiling CUDA kernel (this may take a few seconds)...")
module = cp.RawModule(code=cuda_source, options=('-std=c++11',), name_expressions=('pbkdf2_kernel',))
pbkdf2_kernel = module.get_function('pbkdf2_kernel')

import string
CHARS = string.printable[:-6]
def gen_batch(batch_size, length):
    arr = np.zeros((batch_size, length), dtype=np.uint8)
    for i in range(batch_size):
        s = ''.join(secrets.choice(CHARS) for _ in range(length))
        arr[i,:] = np.frombuffer(s.encode('latin1'), dtype=np.uint8)
    return arr

def gen_salts(batch_size):
    return np.tile(np.frombuffer(salt, dtype=np.uint8), (batch_size,1))

def insert_target_into_batch(batch_np, target_bytes):
    i = secrets.randbelow(batch_np.shape[0])
    b = np.frombuffer(target_bytes, dtype=np.uint8)
    if b.size < batch_np.shape[1]:
        tmp = np.zeros(batch_np.shape[1], dtype=np.uint8)
        tmp[:b.size] = b
        b = tmp
    batch_np[i,:] = b[:batch_np.shape[1]]
    return i
def run():
    found = False
    total_candidate_count = 0
    timings = []
    found_info = None

    for batch_index in range(N_BATCHES):
        batch_np = gen_batch(BATCH_SIZE, L)
        if INSERT_TARGET and batch_index == secrets.randbelow(N_BATCHES):
            pos = insert_target_into_batch(batch_np, target_password.encode())
            print(f"Inserted target into batch {batch_index} at pos {pos}")
        salts_np = gen_salts(BATCH_SIZE)

        d_passwords = cp.asarray(batch_np)
        d_salts = cp.asarray(salts_np)
        d_out = cp.empty((BATCH_SIZE, DKLEN), dtype=cp.uint8)

        blocks = (BATCH_SIZE + TPB - 1) // TPB

        if batch_index == 0:
            pbkdf2_kernel((blocks,), (TPB,), (d_passwords, np.int32(L), d_salts, np.int32(len(salt)), np.int32(1), d_out, np.int32(DKLEN), np.int32(BATCH_SIZE)))
            cp.cuda.runtime.deviceSynchronize()

        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        pbkdf2_kernel((blocks,), (TPB,), (d_passwords, np.int32(L), d_salts, np.int32(len(salt)), np.int32(ITER), d_out, np.int32(DKLEN), np.int32(BATCH_SIZE)))
        end.record()
        end.synchronize()
        elapsed_ms = cp.cuda.get_elapsed_time(start, end)
        elapsed_s = float(elapsed_ms) / 1000.0
        timings.append(elapsed_s)

        out_np = d_out.get()
        for i in range(BATCH_SIZE):
            if bytes(out_np[i,:]) == target_hash:
                found = True
                found_info = (batch_index, i)
                break

        total_candidate_count += BATCH_SIZE
        print(f"[GPU] batch {batch_index} done: time={elapsed_s:.4f}s (cumulative candidates={total_candidate_count})")
        if found:
            print("FOUND on GPU in batch", found_info)
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
    print("GPU run complete. Saved to", OUT_CSV)

if __name__ == "__main__":
    run()
