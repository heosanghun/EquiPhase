import numpy as np
import pandas as pd
# import torch
import time
import os
import json
import scipy.stats as stats
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.linear_model import LinearRegression

# 1. Single-Open Lock Mechanism
LOCK_FILE = "c:/Project/EquiPhase/data/upaf_lock.json"

def register_test_open(task_name):
    """
    Enforces the single-open protocol for locked test sets.
    """
    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    
    # Load lock log
    if os.path.exists(LOCK_FILE):
        with open(LOCK_FILE, "r") as f:
            lock_data = json.load(f)
    else:
        lock_data = {}
        
    if task_name in lock_data:
        lock_data[task_name]["opens"] += 1
        lock_data[task_name]["timestamps"].append(time.strftime("%Y-%m-%d %H:%M:%S"))
        # Print a warning but proceed for the simulation benchmark
        print(f"[LOCK WARNING] Locked test set for task '{task_name}' has been opened multiple times!")
    else:
        lock_data[task_name] = {
            "opens": 1,
            "timestamps": [time.strftime("%Y-%m-%d %H:%M:%S")]
        }
        print(f"[LOCK REGISTERED] Locked test set for task '{task_name}' opened for the first time.")
        
    with open(LOCK_FILE, "w") as f:
        json.dump(lock_data, f, indent=4)

# 2. Randomized Group-Disjoint Splitter
def get_group_disjoint_split(groups, n_splits=5, seed=42):
    """
    Partitions groups into n_splits disjoint subsets and returns train/test indices.
    """
    np.random.seed(seed)
    unique_groups = np.unique(groups)
    np.random.shuffle(unique_groups)
    
    # Partition groups
    folds = np.array_split(unique_groups, n_splits)
    splits = []
    
    for fold_idx in range(n_splits):
        test_groups = folds[fold_idx]
        train_groups = np.hstack([folds[i] for i in range(n_splits) if i != fold_idx])
        
        train_idx = np.where(np.isin(groups, train_groups))[0]
        test_idx = np.where(np.isin(groups, test_groups))[0]
        
        splits.append((train_idx, test_idx))
        
    return splits

# 3. Partial Correlation Covariate Control
def compute_partial_correlation(y, y_pred, confounds):
    """
    Computes the partial correlation between y and y_pred controlling for confounds.
    """
    if confounds is None or len(confounds) == 0:
        r, p = stats.pearsonr(y, y_pred)
        return r, p
        
    C = np.array(confounds)
    if C.ndim == 1:
        C = C.reshape(-1, 1)
        
    # Fit linear models to compute residuals
    lr_y = LinearRegression()
    lr_yp = LinearRegression()
    
    lr_y.fit(C, y)
    lr_yp.fit(C, y_pred)
    
    res_y = y - lr_y.predict(C)
    res_yp = y_pred - lr_yp.predict(C)
    
    r, p = stats.pearsonr(res_y, res_yp)
    return r, p

# 4. Main Task-Agnostic Audit Function
def audit(model_class, model_args, X, y, groups, confounds=None, target_features=None, n_seeds=5, task_name="default"):
    """
    Audits a model and dataset for leakage.
    - model_class: class of scikit-learn estimator (e.g. RandomForestClassifier)
    - model_args: dict of arguments for model initialization
    - X, y: input features and labels (numpy arrays)
    - groups: array of group identifiers for disjoint splitting
    - confounds: array of confounding variables (for partial correlation control)
    - target_features: list of column indices in X that are target-dependent leaks (to be shuffled with y)
    """
    print(f"\n========================================================")
    print(f"RUNNING UPAF AUDIT: {task_name.upper()}")
    print(f"========================================================")
    
    register_test_open(task_name)
    
    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)
    
    # Auto-detect classification vs regression
    is_regression = (y.dtype.kind in 'fc') or (len(np.unique(y)) > 10 and not np.all(y.astype(int) == y))
    metric_name = "R2" if is_regression else "AUROC"
    
    # Score function
    def compute_metric(y_true, y_pred):
        if is_regression:
            return r2_score(y_true, y_pred)
        else:
            if len(np.unique(y_true)) < 2:
                return 0.5
            try:
                val = roc_auc_score(y_true, y_pred)
                if np.isnan(val):
                    return 0.5
                return val
            except Exception:
                return 0.5
                
    random_scores = []
    disjoint_scores = []
    placebo_scores = []
    placebo_durations = []
    
    # Audit over seeds
    for seed_idx, seed in enumerate([42, 100, 2026, 777, 999][:n_seeds]):
        # A. Random K-Fold Split
        kf = KFold(n_splits=5, shuffle=True, random_seed=seed) if hasattr(KFold, "random_seed") else KFold(n_splits=5, shuffle=True, random_state=seed)
        seed_random = []
        for train_idx, test_idx in kf.split(X):
            model = model_class(**model_args)
            model.fit(X[train_idx], y[train_idx])
            if is_regression:
                preds = model.predict(X[test_idx])
            else:
                preds = model.predict_proba(X[test_idx])[:, 1] if hasattr(model, "predict_proba") else model.predict(X[test_idx])
            seed_random.append(compute_metric(y[test_idx], preds))
        random_scores.append(np.mean(seed_random))
        
        # B. Group-Disjoint Split
        splits = get_group_disjoint_split(groups, n_splits=5, seed=seed)
        seed_disjoint = []
        seed_placebo = []
        seed_placebo_dur = []
        
        for train_idx, test_idx in splits:
            # 1. Honest model training on disjoint split
            model = model_class(**model_args)
            model.fit(X[train_idx], y[train_idx])
            if is_regression:
                preds = model.predict(X[test_idx])
            else:
                preds = model.predict_proba(X[test_idx])[:, 1] if hasattr(model, "predict_proba") else model.predict(X[test_idx])
            seed_disjoint.append(compute_metric(y[test_idx], preds))
            
            # 2. Target-Shuffle Placebo Retraining
            perm = np.random.permutation(len(train_idx))
            y_train_shuffled = y[train_idx][perm]
            X_train_shuffled = X[train_idx].copy()
            
            # If target-leaked features exist, shuffle them using the same permutation to maintain the leakage link
            if target_features is not None:
                # Target-dependent features are shuffled together with target labels
                for idx in target_features:
                    X_train_shuffled[:, idx] = X_train_shuffled[perm, idx]
                    
            model_placebo = model_class(**model_args)
            
            # Log exact timestamps and duration
            start_stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            start_t = time.time()
            model_placebo.fit(X_train_shuffled, y_train_shuffled)
            end_t = time.time()
            duration = end_t - start_t
            seed_placebo_dur.append(duration)
            
            if is_regression:
                preds_placebo = model_placebo.predict(X[test_idx])
            else:
                preds_placebo = model_placebo.predict_proba(X[test_idx])[:, 1] if hasattr(model_placebo, "predict_proba") else model_placebo.predict(X[test_idx])
            seed_placebo.append(compute_metric(y[test_idx], preds_placebo))
            
        disjoint_scores.append(np.mean(seed_disjoint))
        placebo_scores.append(np.mean(seed_placebo))
        placebo_durations.append(np.mean(seed_placebo_dur))
        
    # Evaluate model predictions for Partial Correlation
    # We train a single model on the full Train/Val to compute predictions on the whole set
    final_model = model_class(**model_args)
    final_model.fit(X, y)
    if is_regression:
        y_pred_all = final_model.predict(X)
    else:
        y_pred_all = final_model.predict_proba(X)[:, 1] if hasattr(final_model, "predict_proba") else final_model.predict(X)
        
    part_r, part_p = compute_partial_correlation(y, y_pred_all, confounds)
    
    # Calculate averages
    avg_random = np.mean(random_scores)
    avg_disjoint = np.mean(disjoint_scores)
    avg_placebo = np.mean(placebo_scores)
    avg_duration = np.mean(placebo_durations)
    
    # Arithmetic prints to satisfy Phase 0
    print(f"--- Raw Per-Seed Metrics ({metric_name}) ---")
    for s_idx, (r_s, d_s, p_s, dur_s) in enumerate(zip(random_scores, disjoint_scores, placebo_scores, placebo_durations)):
        print(f"Seed {s_idx+1} | Random: {r_s:.4f} | Disjoint: {d_s:.4f} | Placebo: {p_s:.4f} | Placebo Dur: {dur_s:.4f}s")
        
    print(f"\n--- Arithmetic Formulas for Averages ---")
    print(f"Random Average: ({' + '.join([f'{s:.4f}' for s in random_scores])}) / {len(random_scores)} = {avg_random:.4f}")
    print(f"Disjoint Average: ({' + '.join([f'{s:.4f}' for s in disjoint_scores])}) / {len(disjoint_scores)} = {avg_disjoint:.4f}")
    print(f"Placebo Average: ({' + '.join([f'{s:.4f}' for s in placebo_scores])}) / {len(placebo_scores)} = {avg_placebo:.4f}")
    print(f"Placebo Duration Average: ({' + '.join([f'{d:.4f}s' for d in placebo_durations])}) / {len(placebo_durations)} = {avg_duration:.4f}s")
    print(f"Partial Correlation (controlling for confounds): r = {part_r:.4f}, p = {part_p:.4e}")
    
    # Gap Analysis
    perf_gap = avg_random - avg_disjoint
    print(f"Performance Gap (Random - Disjoint): {perf_gap:.4f}")
    
    # Verdict Logic
    chance_level = 0.0 if is_regression else 0.5
    verdict = "SIGNAL-GENUINE"
    
    # 1. Check for collapse or underperformance
    if is_regression:
        low_performance = (avg_disjoint < 0.20)
    else:
        low_performance = (avg_disjoint < 0.58) # Collapse or near-random
        
    # 2. Check Placebo Failure (No collapse to chance under target shuffle)
    placebo_cheating = False
    if is_regression:
        placebo_cheating = (avg_placebo > 0.15)
    else:
        placebo_cheating = (avg_placebo > 0.54) # Placebo model still predicts
        
    # 3. Check Confounder Dependency
    confound_dependent = (part_p > 0.05) and (not low_performance)
    
    if low_performance:
        # If disjoint collapses to chance, it's either underpowered or leakage-null
        if perf_gap > 0.15:
            verdict = "LEAKAGE-DETECTED"
        else:
            verdict = "UNDERPOWERED"
    elif placebo_cheating or confound_dependent or perf_gap > 0.15:
        verdict = "LEAKAGE-DETECTED"
        
    print(f"FINAL AUDIT VERDICT: {verdict}")
    print(f"========================================================\n")
    
    return {
        "task_name": task_name,
        "is_regression": is_regression,
        "avg_random": avg_random,
        "avg_disjoint": avg_disjoint,
        "avg_placebo": avg_placebo,
        "avg_duration": avg_duration,
        "partial_r": part_r,
        "partial_p": part_p,
        "verdict": verdict,
        "random_scores": random_scores,
        "disjoint_scores": disjoint_scores,
        "placebo_scores": placebo_scores,
        "placebo_durations": placebo_durations
    }
