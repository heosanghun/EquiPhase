import os
import sys
import json
import shutil
import hashlib
import numpy as np
import pandas as pd

sys.path.append(r"C:\Project\AI\EquiPhase")
from equiphase.eval.upaf_manifest import UPAFManifest, UPAFEvaluator, IntegrityViolation

print("=" * 80)
print("=== UPAF MANIFEST SPECIFICATION V1: 11 PHYSICAL MUTATION TAMPER TESTS SUITE (T0~T10) ===")
print("=" * 80)

scratch_dir = r"/home/user\.gemini\antigravity\brain\217c14e4-fe62-43ea-af3e-e1ef4c536e8f\scratch"
test_ledger_path = os.path.join(scratch_dir, "test_manifest_ledger_suite.jsonl")
temp_pred_path = os.path.join(scratch_dir, "predictions", "temp_pred.csv")
temp_mod_path = os.path.join(scratch_dir, "temp_module.py")
temp_dataset = os.path.join(scratch_dir, "temp_disk_dataset.csv")

with open(temp_dataset, "w") as f:
    f.write("feat1,label\n1.0,1\n2.0,0\n")

if os.path.exists(test_ledger_path):
    os.remove(test_ledger_path)

with open(temp_mod_path, "w") as f:
    f.write("# Temp module\nVAL = 1\n")

df_pred = pd.DataFrame([{
    "sample_id": "S1", "group_id": "G1", "fold": 0, "seed": 42,
    "split": "test", "y_true": 1, "y_score": 0.9, "y_pred": 1, "protocol": "test"
}])
UPAFEvaluator.save_prediction_persistence(df_pred, temp_pred_path)

test_results = []

def run_test(test_id, manifest_obj, expected_verdict, expect_violation):
    res = UPAFEvaluator.verify_and_update_lock(manifest_obj, test_ledger_path)
    actual_verdict = res["verdict"]
    actual_violation = res["violation"]
    passed = (actual_verdict == expected_verdict and actual_violation == expect_violation)
    test_results.append({
        "test_id": test_id,
        "expected": expected_verdict,
        "actual": actual_verdict,
        "violation": actual_violation,
        "status": "PASS" if passed else "FAIL"
    })
    return res

# Base Manifest Creation Helper
def make_base(task_id="Task_T0", x_val=1.0, f_names=["f1"], splits={"train": [0], "test": [1]}, conf=np.array([[0.5], [0.1]])):
    m = UPAFManifest(task_id=task_id, task_type="binary", entry_script=os.path.abspath(__file__))
    m.seal_data(np.array([[x_val], [2.0]]), np.array([1, 0]), confounds=conf,
                feature_names=f_names, raw_files=[temp_dataset])
    m.seal_splits(splits, "SingleValSplit(n=2)")
    m.seal_code_env(local_module_paths=[temp_mod_path])
    m.seal_execution({"class": "Evaluator"}, [42], {"primary": "roc_auc"})
    m.seal_outputs(predictions_path=temp_pred_path)
    return m

# T0: Initial Seal
m0 = make_base("Task_T0")
run_test("T0_disk_reload", m0, "FIRST_SEAL", False)

# T1: Identical Rerun
m1 = make_base("Task_T0")
run_test("T1_identical_rerun", m1, "REOPEN", False)

# T2: True Physical Array Mutation (X[0,0] += 1e-9)
m2 = make_base("Task_T0", x_val=1.0 + 1e-9)
run_test("T2_single_element_change", m2, "TAMPER", True)

# T3: True Column Swap Mutation
m3 = make_base("Task_T0", f_names=["f2"])
run_test("T3_column_order_swap", m3, "TAMPER", True)

# T4: True Split Indices Mutation
m4 = make_base("Task_T0", splits={"train": [1], "test": [0]})
run_test("T4_split_indices_change", m4, "TAMPER", True)

# T5: True Physical File Modification (Adding comment line to temp_module.py)
with open(temp_mod_path, "a") as f:
    f.write("# Physical modification\n")
m5 = make_base("Task_T0")
run_test("T5_code_modification", m5, "TAMPER", True)
# Restore module file
with open(temp_mod_path, "w") as f:
    f.write("# Temp module\nVAL = 1\n")

# T6: Benign Drift (Runtime version update)
m6 = make_base("Task_T0")
m6.sealed["runtime"]["numpy"] = "9.9.9"
run_test("T6_numpy_version_drift", m6, "BENIGN_DRIFT", False)

# T7: Confound Removal (Tamper)
m7 = make_base("Task_T0", conf=None)
run_test("T7_confound_removal", m7, "TAMPER", True)

# T8: True Physical Prediction CSV File Swap
df_pred_alt = pd.DataFrame([{
    "sample_id": "S1", "group_id": "G1", "fold": 0, "seed": 42,
    "split": "test", "y_true": 1, "y_score": 0.1, "y_pred": 0, "protocol": "test"
}])
UPAFEvaluator.save_prediction_persistence(df_pred_alt, temp_pred_path)
m8 = make_base("Task_T0")
run_test("T8_prediction_file_swap", m8, "TAMPER", True)
# Restore pred file
UPAFEvaluator.save_prediction_persistence(df_pred, temp_pred_path)

# T9: True Physical Ledger File Record Tamper Detection
with open(test_ledger_path, "r") as f:
    lines = f.readlines()
lines[0] = lines[0].replace('"open_count": 1', '"open_count": 99')
with open(test_ledger_path, "w") as f:
    f.writelines(lines)
m9 = make_base("Task_T0")
run_test("T9_ledger_file_edit", m9, "LEDGER_RECORD_TAMPERED", True)

# T10: True Ledger Chain Break Detection (Deleting Intermediate Line after GENESIS)
if os.path.exists(test_ledger_path):
    os.remove(test_ledger_path)

m_c1 = make_base("Task_Chain")
res_c1 = UPAFEvaluator.verify_and_update_lock(m_c1, test_ledger_path)

# Append GENESIS record
with open(test_ledger_path, "r") as f:
    lines_c = f.readlines()
last_rec = json.loads(lines_c[-1])
gen_rec = {
    "manifest_version": "1.0",
    "task_id": "Task_Chain",
    "seal_status": "GENESIS",
    "prev_manifest_self_sha256": last_rec["manifest_self_sha256"]
}
gen_rec["manifest_self_sha256"] = hashlib.sha256(json.dumps(gen_rec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
with open(test_ledger_path, "a") as f:
    f.write(json.dumps(gen_rec, sort_keys=True, ensure_ascii=False) + "\n")

# Run record 2 after GENESIS
m_c2 = make_base("Task_Chain")
m_c2.sealed["runtime"]["numpy"] = "9.9.9"
res_c2 = UPAFEvaluator.verify_and_update_lock(m_c2, test_ledger_path)

# Tamper chain: delete intermediate GENESIS line (line index 2)
with open(test_ledger_path, "r") as f:
    lines_c = f.readlines()
del lines_c[1] # Delete line 2
with open(test_ledger_path, "w") as f:
    f.writelines(lines_c)

m_c3 = make_base("Task_Chain")
run_test("T10_chain_break", m_c3, "LEDGER_CHAIN_BROKEN", True)

print(f"{'Test ID':<28} | {'Expected':<22} | {'Actual Verdict':<24} | {'Violation':<10} | {'Status':<8}")
print("-" * 100)
for r in test_results:
    print(f"{r['test_id']:<28} | {r['expected']:<22} | {r['actual']:<24} | {str(r['violation']):<10} | {r['status']:<8}")

all_pass = all(r["status"] == "PASS" for r in test_results)
print("=" * 100)
print(f"TAMPER SUITE RESULT: {sum(1 for r in test_results if r['status'] == 'PASS')} / {len(test_results)} PASSED.")
print("=" * 100)
