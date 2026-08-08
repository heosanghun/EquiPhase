import subprocess
import os

repo = r"C:\Project\EquiPhase"

def exec_print(cmd_str):
    print(f"\n$ {cmd_str}")
    p = subprocess.run(cmd_str, shell=True, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    print(p.stdout.strip())

exec_print("certutil -hashfile base_det1.txt SHA256")
exec_print("certutil -hashfile base_det2.txt SHA256")
exec_print("powershell -Command \"Get-Content base_det1.txt -Tail 3\"")
exec_print("powershell -Command \"Get-Content base_det2.txt -Tail 3\"")
exec_print("powershell -Command \"$f1 = Get-Content base_det1.txt | Where-Object { $_ -notmatch 'WALL-CLOCK' -and $_ -notmatch 'mtime' }; $f2 = Get-Content base_det2.txt | Where-Object { $_ -notmatch 'WALL-CLOCK' -and $_ -notmatch 'mtime' }; Compare-Object $f1 $f2; 'COMPARE_DONE'\"")
