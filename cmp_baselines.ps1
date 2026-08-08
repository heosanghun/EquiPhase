$f1 = Get-Content C:\Project\EquiPhase\base_cpu_run1.txt | Where-Object { $_ -notmatch 'WALL-CLOCK' -and $_ -notmatch 'mtime' }
$f2 = Get-Content C:\Project\EquiPhase\base_cpu_run2.txt | Where-Object { $_ -notmatch 'WALL-CLOCK' -and $_ -notmatch 'mtime' }
$diff = Compare-Object $f1 $f2
if ($null -eq $diff) {
    Write-Host "COMPARE-OBJECT CPU RESULT: ZERO DIFF (EXACT BITWISE MATCH!)"
} else {
    Write-Host "COMPARE-OBJECT CPU RESULT: DISCREPANCY FOUND"
    $diff | Format-Table -AutoSize
}
