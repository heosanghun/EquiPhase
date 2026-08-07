import os
import sys

sys.path.append(r"C:\Project\AI\EquiPhase")
from equiphase.eval.upaf_manifest import UPAFEvaluator

print("=" * 100)
print("=== NON-MUTATING PURE VERIFY_ONLY INSPECTION OF REPOSITORY OPERATIONAL LEDGER ===")
print("=" * 100)

repo_ledger = r"C:\Project\AI\EquiPhase\ledger\operational_manifest_ledger.jsonl"
repo_tip = r"C:\Project\AI\EquiPhase\ledger\ledger_tip.sha256"

# Run verify_only
res = UPAFEvaluator.verify_only(repo_ledger, repo_tip)

print(f"Overall Ledger Validity: {res['valid']}")
print(f"Tip SHA-256 Anchor: {res['tip_sha256']}\n")
print(f"{'Line':<6} | {'Task ID':<26} | {'Seal Status':<24} | {'Verification Status'}")
print("-" * 80)
for r in res["line_reports"]:
    if "line" in r and isinstance(r["line"], int):
        print(f"Line {r['line']:>2d} | {r.get('task_id', 'N/A'):<26} | {r.get('status', 'N/A'):<24} | {r['verification']}")
    else:
        print(f"Anchor | Verification: {r.get('verification')} | Anchored: {r.get('anchored')} | Actual: {r.get('actual')}")

print("=" * 100)
