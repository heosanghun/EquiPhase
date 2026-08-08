import os

# Set environment variables for CPU determinism
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import subprocess
import sys

def run_instance(outfile):
    print(f"Starting execution for {outfile}...")
    cmd = [sys.executable, r"C:\Project\EquiPhase\claude_paper2_baselines_sealed.py"]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    with open(outfile, "w", encoding="utf-8") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True, env=env)
    print(f"Finished {outfile} with return code {res.returncode}")
    if res.stderr:
        print(f"Stderr: {res.stderr[:200]}")

run_instance(r"C:\Project\EquiPhase\base_cpu1.txt")
run_instance(r"C:\Project\EquiPhase\base_cpu2.txt")

print("Both CPU runs complete. Checking diff...")
with open(r"C:\Project\EquiPhase\base_cpu1.txt", "r", encoding="utf-8") as f:
    l1 = [line for line in f.read().splitlines() if "WALL-CLOCK" not in line and "mtime" not in line]
with open(r"C:\Project\EquiPhase\base_cpu2.txt", "r", encoding="utf-8") as f:
    l2 = [line for line in f.read().splitlines() if "WALL-CLOCK" not in line and "mtime" not in line]

diffs = [i for i, (a, b) in enumerate(zip(l1, l2)) if a != b]
print(f"Total diff lines: {len(diffs)} / {len(l1)}")
if len(diffs) == 0 and len(l1) == len(l2):
    print("=== SUCCESS: CPU RUN 1 AND CPU RUN 2 ARE 100% BITWISE IDENTICAL ===")
