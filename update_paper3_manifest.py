import os
import hashlib

paper3_dir = "paper3_iclr"
manifest_path = os.path.join(paper3_dir, "PAPER3_MANIFEST_20260808.txt")

files = sorted([f for f in os.listdir(paper3_dir) if os.path.isfile(os.path.join(paper3_dir, f)) and f != "PAPER3_MANIFEST_20260808.txt"])

lines = ["PAPER3_ICLR SHA-256 MANIFEST\n", "="*80 + "\n"]
for f in files:
    fpath = os.path.join(paper3_dir, f)
    h = hashlib.sha256(open(fpath, "rb").read()).hexdigest().upper()
    lines.append(f"{h}  {f}\n")

with open(manifest_path, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"Updated manifest at {manifest_path}:")
print("".join(lines))
