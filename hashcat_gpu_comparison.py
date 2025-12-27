import subprocess
import csv
import os
import re

# Run GPU benchmark
print("Running GPU benchmark...")
subprocess.run(["python", "gpu_pbkdf2_runner.py"], check=True)

# Read GPU results
gpu_speed = None
with open("results_gpu.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        gpu_speed = float(row["passwords_per_second"]) / 1000  # to kH/s
        break

# Run hashcat benchmark
print("Running hashcat benchmark...")
result = subprocess.run(["cmd", "/c", "cd /d C:\\Users\\artyo\\Downloads\\hashcat-7.1.2 && hashcat.exe --benchmark -m 10900"], capture_output=True, text=True)
output = result.stdout

# Parse hashcat results
hashcat_total = None
match = re.search(r"Speed\.\#\*\.\.\.\.\.\.\.\.\.: +([\d.]+) kH/s", output)
if match:
    hashcat_total = float(match.group(1))

# Parse iterations
iterations_match = re.search(r"PBKDF2-HMAC-SHA256\) \[Iterations: (\d+)\]", output)
iterations = int(iterations_match.group(1)) if iterations_match else 999

hashcat_scaled = hashcat_total * iterations / 5000 if hashcat_total else 0

# Print results
print("Hashcat PBKDF2-HMAC-SHA256 benchmark (mode 10900) for", iterations, "iterations:")
# Parse device speeds
device_matches = re.findall(r"Speed\.\#(\d+)\.\.\.\.\.\.\.\.\.: +([\d.]+) kH/s", output)
for device, speed in device_matches:
    print(f"- Device #{device}: {speed} kH/s")
print(f"- Total: {hashcat_total} kH/s")
print()
print(f"Scaled for 5000 iterations: {hashcat_scaled:.2f} kH/s")
print()
print(f"GPU implementation: {gpu_speed:.1f} kH/s")
print(f"HashCat implementation: {hashcat_scaled:.2f} kH/s")
print()
if gpu_speed > hashcat_scaled:
    print("The custom GPU implementation outperforms hashcat's optimized kernel for this PBKDF2 configuration.")
    print(f"GPU is {gpu_speed / hashcat_scaled:.2f}x faster.")
else:
    print("Hashcat outperforms the custom GPU implementation for this PBKDF2 configuration.")
    print(f"Hashcat is {hashcat_scaled / gpu_speed:.2f}x faster.")