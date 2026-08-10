import os
import re

files = ["run1_raw.txt", "run2_raw.txt", "run3_raw.txt"]
repo = r"/home/user/EquiPhase"

print("=== FILE INTEGRITY CHECK ===")
for fname in files:
    fpath = os.path.join(repo, fname)
    if not os.path.exists(fpath):
        print(f"{fname}: MISSING")
        continue
    size = os.path.getsize(fpath)
    with open(fpath, "rb") as f:
        data = f.read()
    try:
        text = data.decode("utf-16le")
    except Exception:
        text = data.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    print(f"File: {fname}")
    print(f"  Byte Size: {size}")
    print(f"  Line Count: {len(lines)}")
    print(f"  Tail 3 Lines:")
    for line in lines[-3:]:
        print(f"    {line}")
    print("-" * 50)

# Filter out WALL-CLOCK and mtime
def filter_lines(fname):
    fpath = os.path.join(repo, fname)
    with open(fpath, "rb") as f:
        data = f.read()
    try:
        text = data.decode("utf-16le")
    except Exception:
        text = data.decode("utf-8", errors="ignore")
    filtered = []
    for line in text.splitlines():
        if "WALL-CLOCK" in line or "mtime" in line:
            continue
        filtered.append(line)
    return filtered

f1 = filter_lines("run1_raw.txt")
f2 = filter_lines("run2_raw.txt")

print("=== DIFF (run1_raw vs run2_raw, excluding WALL-CLOCK and mtime) ===")
diffs = []
for idx, (l1, l2) in enumerate(zip(f1, f2)):
    if l1 != l2:
        diffs.append((idx, l1, l2))

if len(f1) != len(f2):
    print(f"Line length mismatch: run1_raw={len(f1)} vs run2_raw={len(f2)}")
elif len(diffs) == 0:
    print("DIFF RESULT: EXACT MATCH (0 differences found!)")
else:
    print(f"DIFF RESULT: {len(diffs)} differences found")
    for idx, l1, l2 in diffs[:10]:
        print(f"Line {idx}:")
        print(f"  run1: {l1}")
        print(f"  run2: {l2}")

print("=" * 50)
print("=== SEC 7 BLOCK FROM run3_raw.txt ===")
f3path = os.path.join(repo, "run3_raw.txt")
with open(f3path, "rb") as f:
    data3 = f.read()
try:
    text3 = data3.decode("utf-16le")
except Exception:
    text3 = data3.decode("utf-8", errors="ignore")

sec7_lines = []
in_sec7 = False
for line in text3.splitlines():
    if "[SEC 7]" in line or "SEC 7" in line:
        in_sec7 = True
    if in_sec7:
        sec7_lines.append(line)

print("\n".join(sec7_lines))

print("=" * 50)
print("=== SEC 5 CROSS-GENERATION CHECK (run1_utf8 vs run1_raw) ===")
def get_sec5(fname):
    fpath = os.path.join(repo, fname)
    with open(fpath, "rb") as f:
        data = f.read()
    try:
        text = data.decode("utf-16le")
    except Exception:
        text = data.decode("utf-8", errors="ignore")
    sec5 = []
    in_sec = False
    for line in text.splitlines():
        if "[SEC 5]" in line or "[SEC5" in line:
            in_sec = True
        elif "[SEC 6]" in line:
            break
        if in_sec:
            sec5.append(line)
    return sec5

s5_old = get_sec5("audit_run1.txt")
s5_new = get_sec5("run1_raw.txt")

print(f"Old audit_run1.txt SEC 5 lines: {len(s5_old)}")
print(f"New run1_raw.txt   SEC 5 lines: {len(s5_new)}")
diffs = [(i, a, b) for i, (a, b) in enumerate(zip(s5_old, s5_new)) if a != b]
if len(s5_old) == len(s5_new) and len(diffs) == 0:
    print("SEC 5 CROSS-GENERATION CHECK: EXACT BIT-FOR-BIT MATCH (0 diffs)!")
else:
    print(f"Differences count: {len(diffs)}")
    for i, a, b in diffs[:5]:
        print(f"  Line {i}: Old='{a}' vs New='{b}'")
