import subprocess
import os

repo = r"C:\Project\EquiPhase"

def run_cmd(cmd, label):
    print(f"\n========================================================================================")
    print(f"=== {label} ===")
    print(f"=== COMMAND: {cmd} ===")
    print(f"========================================================================================")
    res = subprocess.run(cmd, shell=True, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    print(res.stdout)

run_cmd("certutil -hashfile base_det1.txt SHA256", "COMMAND 1: base_det1.txt SHA256")
run_cmd("certutil -hashfile base_det2.txt SHA256", "COMMAND 2: base_det2.txt SHA256")
run_cmd("powershell -Command \"Get-Content base_det1.txt -Tail 3\"", "COMMAND 3a: base_det1.txt Tail 3")
run_cmd("powershell -Command \"Get-Content base_det2.txt -Tail 3\"", "COMMAND 3b: base_det2.txt Tail 3")
run_cmd("powershell -Command \"$f1 = Get-Content base_det1.txt | Where-Object { $_ -notmatch 'WALL-CLOCK' -and $_ -notmatch 'mtime' }; $f2 = Get-Content base_det2.txt | Where-Object { $_ -notmatch 'WALL-CLOCK' -and $_ -notmatch 'mtime' }; Compare-Object $f1 $f2; 'COMPARE_DONE'\"", "COMMAND 4: Compare-Object base_det1 vs base_det2")
