import os
import hashlib

manifest_file = os.path.join("paper3_iclr", "PAPER3_MANIFEST_20260808.txt")

with open(manifest_file, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip() and not line.startswith("PAPER3_ICLR") and not line.startswith("=")]

print("=== VERIFYING PAPER3_ICLR BITWISE SHA-256 MANIFEST ===")
all_pass = True
for line in lines:
    parts = line.split(maxsplit=1)
    if len(parts) != 2:
        continue
    expected_hash, fname = parts
    fpath = os.path.join("paper3_iclr", fname)
    if not os.path.exists(fpath):
        print(f"FAIL: File missing -> {fname}")
        all_pass = False
        continue
    actual_hash = hashlib.sha256(open(fpath, "rb").read()).hexdigest().upper()
    if actual_hash == expected_hash:
        print(f"[PASS] {fname}: SHA256 matches ({actual_hash[:8]}...)")
    else:
        print(f"[FAIL] {fname}: Hash mismatch! Expected {expected_hash}, got {actual_hash}")
        all_pass = False

if all_pass:
    print("\n>>> ALL 10 MANIFEST FILES VERIFIED 100% PASS <<<")
else:
    print("\n>>> MANIFEST VERIFICATION FAILED <<<")
