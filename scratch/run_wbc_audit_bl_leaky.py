import os
import sys
import pickle
import numpy as np

# Ensure workspace is in path
sys.path.append("c:/Project/EquiPhase")

import upaf
from sklearn.ensemble import RandomForestClassifier

def main():
    print("========================================================")
    print("WBCBENCH 2026 PATIENT-LEVEL LEAKAGE AUDIT (ONLY LEAKY BG)")
    print("========================================================\n")
    
    # Load extracted features
    pickle_path = "data/wbc_features_bl.pkl"
    if not os.path.exists(pickle_path):
        print(f"Error: {pickle_path} not found. Please run extract_wbc_features_bl.py first.")
        sys.exit(1)
        
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
        
    # ONLY USE THE FIRST 3 FEATURES: Corner background colors (avg_bg)
    X = data["features"][:, :3]
    y = data["labels"]
    groups = data["groups"]
    
    print(f"Loaded {len(X)} samples.")
    print(f"Features shape: {X.shape} (Only background R, G, B)")
    print(f"Class 1 (BL) count: {np.sum(y == 1)} ({np.mean(y == 1)*100:.2f}%)")
    print(f"Unique patients (groups): {len(np.unique(groups))}")
    
    # Run UPAF audit using RandomForestClassifier
    # Since we only use background features, we expect leakage to be detected:
    # Random Split AUROC > 0.60, Disjoint Split AUROC ~ 0.50.
    audit_res = upaf.audit(
        model_class=RandomForestClassifier,
        model_args={"n_estimators": 50, "max_depth": 5, "random_state": 42},
        X=X,
        y=y,
        groups=groups,
        confounds=None,
        target_features=None,
        n_seeds=5,
        task_name="wbc_patient_leakage_bl_leaky"
    )
    
    # Save audit results
    save_path = "data/wbc_audit_results_bl_leaky.json"
    import json
    with open(save_path, "w") as f:
        json.dump(audit_res, f, indent=4)
    print(f"WBC BL Leaky Audit results saved to {save_path}")

if __name__ == "__main__":
    main()
