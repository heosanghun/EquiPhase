Set-Location C:\Project\EquiPhase

Write-Host "=== COMMAND 1: certutil -hashfile base_det1.txt SHA256 ==="
certutil -hashfile base_det1.txt SHA256

Write-Host ""
Write-Host "=== COMMAND 2: certutil -hashfile base_det2.txt SHA256 ==="
certutil -hashfile base_det2.txt SHA256

Write-Host ""
Write-Host "=== COMMAND 3: Tail 3 of base_det1.txt & base_det2.txt ==="
Write-Host "--- base_det1.txt Tail 3 ---"
Get-Content base_det1.txt -Tail 3
Write-Host "--- base_det2.txt Tail 3 ---"
Get-Content base_det2.txt -Tail 3

Write-Host ""
Write-Host "=== COMMAND 4: Compare-Object base_det1 vs base_det2 (excluding WALL-CLOCK/mtime) ==="
$f1 = Get-Content base_det1.txt | Where-Object { $_ -notmatch 'WALL-CLOCK|mtime' }
$f2 = Get-Content base_det2.txt | Where-Object { $_ -notmatch 'WALL-CLOCK|mtime' }
$diff = Compare-Object $f1 $f2
if ($null -eq $diff) {
    Write-Host "(Compare-Object output is empty)"
} else {
    $diff | Format-Table -AutoSize
}
Write-Host "COMPARE_DONE"
