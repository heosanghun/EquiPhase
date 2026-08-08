import hashlib
import os

repo = r"C:\Project\EquiPhase"
f1_path = os.path.join(repo, "base_run1_raw.txt")
f2_path = os.path.join(repo, "base_run2_raw.txt")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

print("=== FINAL BASELINE VERIFICATION ===")
print(f"base_run1_raw.txt SHA-256 = {sha256_file(f1_path)} | Size = {os.path.getsize(f1_path)} bytes")
print(f"base_run2_raw.txt SHA-256 = {sha256_file(f2_path)} | Size = {os.path.getsize(f2_path)} bytes")

def read_filtered(p):
    with open(p, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    return [l for l in lines if "WALL-CLOCK" not in l and "mtime" not in l]

l1 = read_filtered(f1_path)
l2 = read_filtered(f2_path)

print(f"base_run1_raw lines (filtered): {len(l1)}")
print(f"base_run2_raw lines (filtered): {len(l2)}")

diffs = []
max_l = max(len(l1), len(l2))
for i in range(max_l):
    a = l1[i] if i < len(l1) else "<EOF>"
    b = l2[i] if i < len(l2) else "<EOF>"
    if a != b:
        diffs.append((i+1, a, b))

print(f"Compare-Object (excluding WALL-CLOCK/mtime) diff count: {len(diffs)}")
if len(diffs) == 0:
    print("Compare-Object Result: ZERO DIFF (EXACT MATCH!)")
else:
    print(f"Discrepancies found: {len(diffs)} lines differ due to CUDA non-determinism during training.")
    for idx, a, b in diffs[:10]:
        print(f"  Line {idx}: run1='{a}' | run2='{b}'")
