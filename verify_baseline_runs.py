import os

files = ["base_run1_raw.txt", "base_run2_raw.txt"]
repo = r"/home/user/EquiPhase"

print("=== BASELINE FILE INTEGRITY CHECK ===")
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

f1 = filter_lines("base_run1_raw.txt")
f2 = filter_lines("base_run2_raw.txt")

print("=" * 50)
print("=== BASELINE SUMMARY REPORT (base_run1_raw.txt) ===")
with open(os.path.join(repo, "base_run1_raw.txt"), "rb") as f:
    d1 = f.read()
try: t1 = d1.decode("utf-16le")
except: t1 = d1.decode("utf-8", errors="ignore")

for line in t1.splitlines():
    if any(k in line for k in ["BASELINE", "summary", "N_endpoint", "ratio =", "c =", "effective", "TOTAL"]):
        print(f"  {line}")
