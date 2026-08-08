import os

f1_path = r"C:\Project\EquiPhase\base_det1.txt"
f2_path = r"C:\Project\EquiPhase\base_det2.txt"

def read_filtered(p):
    with open(p, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    return [l for l in lines if "WALL-CLOCK" not in l and "mtime" not in l]

l1 = read_filtered(f1_path)
l2 = read_filtered(f2_path)

print(f"base_cpu_run1 line count (filtered): {len(l1)}")
print(f"base_cpu_run2 line count (filtered): {len(l2)}")

diffs = []
max_l = max(len(l1), len(l2))
for i in range(max_l):
    a = l1[i] if i < len(l1) else "<EOF>"
    b = l2[i] if i < len(l2) else "<EOF>"
    if a != b:
        diffs.append((i+1, a, b))

if len(diffs) == 0:
    print("=== COMPARE RESULT: 100% ZERO DIFF (EXACT MATCH!) ===")
else:
    print(f"=== COMPARE RESULT: {len(diffs)} DIFFERENCES FOUND ===")
    for idx, a, b in diffs[:10]:
        print(f"Line {idx}:")
        print(f"  run1: {a}")
        print(f"  run2: {b}")
