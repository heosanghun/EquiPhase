import subprocess
import os

repo = r"C:\Project\EquiPhase"

def run_and_print(cmd, title):
    print(f"\n========================================================================================")
    print(f"=== {title} ===")
    print(f"========================================================================================")
    res = subprocess.run(cmd, shell=True, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    print(res.stdout.strip())

run_and_print("certutil -hashfile base_det1.txt SHA256", "COMMAND 1: certutil -hashfile base_det1.txt SHA256")
run_and_print("certutil -hashfile base_det2.txt SHA256", "COMMAND 2: certutil -hashfile base_det2.txt SHA256")
run_and_print("powershell -Command \"Get-Content base_det1.txt -Tail 3\"", "COMMAND 3a: Get-Content base_det1.txt -Tail 3")
run_and_print("powershell -Command \"Get-Content base_det2.txt -Tail 3\"", "COMMAND 3b: Get-Content base_det2.txt -Tail 3")
