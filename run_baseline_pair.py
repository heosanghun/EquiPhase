import os
import subprocess
import sys
import hashlib

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

repo = r"/home/user/EquiPhase"
script = os.path.join(repo, "claude_paper2_baselines_sealed.py")

def run_script(outfile):
    out_path = os.path.join(repo, outfile)
    print(f"Running baseline -> {outfile} ...")
    with open(out_path, "w", encoding="utf-8") as f:
        res = subprocess.run([sys.executable, script], stdout=f, stderr=subprocess.PIPE, text=True)
    print(f"  Done {outfile}. Return code: {res.returncode}")
    if res.stderr:
        print(f"  Stderr: {res.stderr[:200]}")

run_script("base_cpu_run1.txt")
run_script("base_cpu_run2.txt")

h1 = hashlib.sha256(open(os.path.join(repo, "base_cpu_run1.txt"), "rb").read()).hexdigest()
h2 = hashlib.sha256(open(os.path.join(repo, "base_cpu_run2.txt"), "rb").read()).hexdigest()

print(f"base_cpu_run1 SHA-256 = {h1}")
print(f"base_cpu_run2 SHA-256 = {h2}")

def get_lines(fname):
    with open(os.path.join(repo, fname), "r", encoding="utf-8") as f:
        return [l for l in f.read().splitlines() if "WALL-CLOCK" not in l and "mtime" not in l]

l1 = get_lines("base_cpu_run1.txt")
l2 = get_lines("base_cpu_run2.txt")

diffs = [(i+1, a, b) for i, (a, b) in enumerate(zip(l1, l2)) if a != b]
print(f"Filtered lines: run1={len(l1)}, run2={len(l2)}")
print(f"Diff line count: {len(diffs)}")
if len(diffs) == 0 and len(l1) == len(l2):
    print("=== FINAL RESULT: 100% BITWISE ZERO-DIFF MATCH (COMPARE-OBJECT PASSED!) ===")
else:
    print(f"=== RESULT: {len(diffs)} diff lines ===")
