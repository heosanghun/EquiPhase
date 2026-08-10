import os
import subprocess
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

repo = r"/home/user/EquiPhase"
script = os.path.join(repo, "claude_paper2_baselines_sealed.py")
outfile = os.path.join(repo, "base_run2_raw.txt")

print(f"Starting unbuffered run to {outfile}...")
with open(outfile, "w", encoding="utf-8", buffering=1) as f:
    res = subprocess.run([sys.executable, script], stdout=f, stderr=subprocess.PIPE, text=True)
print(f"Finished base_run2_raw.txt with returncode {res.returncode}")
if res.stderr:
    print(f"Stderr: {res.stderr[:200]}")
