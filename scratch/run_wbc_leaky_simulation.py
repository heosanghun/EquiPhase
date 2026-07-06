import os
import sys
import pickle
import numpy as np
import pandas as pd

# Ensure workspace is in path
sys.path.append("c:/Project/EquiPhase")

import upaf
from sklearn.ensemble import RandomForestClassifier

def main():
    print("========================================================")
    print("UPAF LEAKAGE SIMULATION ON WBCBENCH 2026 DATASET")
    print("========================================================\n")
    
    # Load labels
    df = pd.read_csv("data/wbc-bench-2026/phase1_label.csv")
    df['binary_label'] = (df['labels'] == 'BL').astype(int)
    df['patient_id'] = df['ID'].apply(lambda x: x[:5])
    
    y = df['binary_label'].values
    groups = df['patient_id'].values
    
    # 1. Generate a genuine cell-level feature X_0: weakly correlated with y
    # We add normal noise to the target y to create a feature with AUROC ~ 0.70
    np.random.seed(42)
    X_0 = y * 0.5 + np.random.randn(len(y)) * 1.0
    
    # 2. Generate a patient-level leaky feature X_1: patient-mean of y
    # This represents patient-level base rate leakage (demographic/clinic prior leakage)
    patient_means = df.groupby('patient_id')['binary_label'].mean().to_dict()
    X_1 = df['patient_id'].map(patient_means).values
    
    # Combine features: Column 0 is the genuine feature, Column 1 is the patient-level leak
    X = np.column_stack([X_0, X_1])
    
    print(f"Loaded {len(X)} samples.")
    print(f"Class 1 (BL) count: {np.sum(y == 1)} ({np.mean(y == 1)*100:.2f}%)")
    print(f"Unique patients (groups): {len(np.unique(groups))}")
    
    # Run UPAF audit using RandomForestClassifier
    # We expect UPAF to detect leakage because X_1 is a group-level leak.
    # Random Split will achieve very high AUROC (~0.90+), while
    # Disjoint Split will only achieve the genuine feature AUROC (~0.70).
    # Performance Gap will be > 0.15 -> LEAKAGE-DETECTED!
    audit_res = upaf.audit(
        model_class=RandomForestClassifier,
        model_args={"n_estimators": 50, "max_depth": 5, "random_state": 42},
        X=X,
        y=y,
        groups=groups,
        confounds=None,
        target_features=[1], # Column 1 (X_1) is target-dependent leak
        n_seeds=5,
        task_name="wbc_patient_leakage_simulation"
    )
    
    # Save audit results
    save_path = "data/wbc_audit_results_simulation.json"
    import json
    with open(save_path, "w") as f:
        json.dump(audit_res, f, indent=4)
    print(f"WBC Simulation Audit results saved to {save_path}")

if __name__ == "__main__":
    main()
