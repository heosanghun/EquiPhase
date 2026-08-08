import subprocess
import hashlib
import os

repo = r"C:\Project\EquiPhase"

files = [
    "paper1_manuscript_draft.md",
    "paper1_submission_format.tex",
    "paper2_manuscript_draft.md",
    "cognios_ir_audit_asset.md",
    "audit_incident_log.md",
    "FREEZE_PAPER2.md"
]

print("========================================================================================")
print("=== 1. FILE EXISTENCE & SHA-256 CHECKSUM VERIFICATION ===")
print("========================================================================================")

for fname in files:
    fpath = os.path.join(repo, fname)
    if os.path.exists(fpath):
        size = os.path.getsize(fpath)
        with open(fpath, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        print(f"[EXIST] {fname:<30} | Size: {size:>6} bytes | SHA256: {sha}")
    else:
        print(f"[MISSING] {fname:<28}")

print("\n========================================================================================")
print("=== 2. GIT COMMIT HISTORY CHAIN (git log -n 6 --oneline) ===")
print("========================================================================================")
res_log = subprocess.run("git log -n 6 --oneline", cwd=repo, shell=True, stdout=subprocess.PIPE, text=True).stdout
print(res_log.strip())

print("\n========================================================================================")
print("=== 3. GIT WORKING TREE CLEANLINESS STATUS (git status) ===")
print("========================================================================================")
res_stat = subprocess.run("git status", cwd=repo, shell=True, stdout=subprocess.PIPE, text=True).stdout
print(res_stat.strip())
