import numpy as np
import pandas as pd
import sys
import os
import json

# Ensure workspace is in path
sys.path.append("D:/AI/EquiPhase")

import upaf
from sklearn.ensemble import RandomForestClassifier

def run_calibration():
    print("========================================================")
    print("PHASE 2: SYNTHETIC LEAKAGE STRESS-TEST BENCHMARK")
    print("========================================================\n")
    
    np.random.seed(42)
    n_samples = 600
    n_groups = 50
    samples_per_group = n_samples // n_groups
    
    # Generate group labels
    groups = np.repeat(np.arange(n_groups), samples_per_group)
    
    # 1. True feature X_0
    X_0 = np.random.randn(n_samples)
    
    # Target label: y = 1 if X_0 > 0 else 0
    y = (X_0 > 0).astype(int)
    
    # Sweeping alpha
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    calibration_results = []
    
    # We will save the results to compile a calibration table
    for alpha in alphas:
        print(f"--- Running Audit with Leakage Strength alpha = {alpha} ---")
        
        # Leaked feature X_1: mixtures of noise and target y
        epsilon = np.random.randn(n_samples)
        # Shift epsilon to make it look like a standard continuous feature
        X_1 = (1.0 - alpha) * epsilon + alpha * (y * 2.0 - 1.0)
        
        # Generate dummy noise features X_2 to X_6
        X_dummy = np.random.randn(n_samples, 5)
        
        # Combine to feature matrix X
        X = np.column_stack([X_0, X_1, X_dummy])
        
        # Run UPAF audit
        # Feature index 1 (X_1) is the target-leaked feature, so it shuffles with y in placebo
        audit_res = upaf.audit(
            model_class=RandomForestClassifier,
            model_args={"n_estimators": 30, "max_depth": 5, "random_state": 42},
            X=X,
            y=y,
            groups=groups,
            confounds=X_1, # Treat the leaked feature as the confounder to test control
            target_features=[1], # X_1 is target-dependent
            n_seeds=5,
            task_name=f"synthetic_alpha_{alpha}"
        )
        
        calibration_results.append({
            "alpha": alpha,
            "avg_random": audit_res["avg_random"],
            "avg_disjoint": audit_res["avg_disjoint"],
            "avg_placebo": audit_res["avg_placebo"],
            "partial_r": audit_res["partial_r"],
            "partial_p": audit_res["partial_p"],
            "verdict": audit_res["verdict"]
        })
        
    # Print the calibration table
    print("\n========================================================")
    print("SYNTHETIC BENCHMARK CALIBRATION RESULTS SUMMARY")
    print("========================================================")
    print("alpha | Random AUROC | Disjoint AUROC | Placebo AUROC | Partial Corr p-val | Verdict")
    print("------+--------------+----------------+---------------+--------------------+-----------------")
    for res in calibration_results:
        print(f"{res['alpha']:.1f}   | {res['avg_random']:.4f}       | {res['avg_disjoint']:.4f}         | {res['avg_placebo']:.4f}        | {res['partial_p']:.4e}         | {res['verdict']}")
    print("========================================================\n")
    
    # Save results to data directory
    save_path = "D:/AI/EquiPhase/data/upaf_calibration_results.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(calibration_results, f, indent=4)
    print(f"Calibration results successfully saved to {save_path}")

if __name__ == "__main__":
    run_calibration()
