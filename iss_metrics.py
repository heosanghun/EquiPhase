import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

def find_critical_lambdas(model, padded_X, device, lam_min=-2.0, lam_max=2.0, num_steps=50, mut_indices=None, X_wt_esm=None):
    """
    Finds the predicted critical lambda* for each sequence in the batch using the learned relation
    from the switch loss baseline: lambda* = -4.0 * (m1 - m0).
    No target structures (D_true) are used or required.
    """
    model.eval()
    B_size = padded_X.shape[0]
    
    if mut_indices is not None:
        mut_indices = mut_indices.to(device)
    if X_wt_esm is not None:
        X_wt_esm = X_wt_esm.to(device)
        
    with torch.no_grad():
        lam_zero = torch.zeros(B_size, 1, device=device)
        # Forward pass at lambda=0 to resolve fixed points and margins
        _, margins_zero, _ = model(padded_X.to(device), lam_zero, mut_indices=mut_indices, X_wt_esm=X_wt_esm)
        
    # Since State A is mapped to start 0 and State B to start 1,
    # signed_delta_m_zero = margins_zero[:, 1] - margins_zero[:, 0]
    delta_m_zero = margins_zero[:, 1] - margins_zero[:, 0]
    pred_lams = (-4.0 * delta_m_zero).cpu().numpy()
        
    return pred_lams, pred_lams

def compute_metrics(pred_critical_lams, pred_stability_diffs, true_ddgs):
    """
    Computes Pearson and Spearman correlations, and AUROC for fold reversal classification.
    - pred_critical_lams: Array of shape (N,) predicted transition points.
    - pred_stability_diffs: Array of shape (N,) predicted stability difference at baseline lambda=0.
    - true_ddgs: Array of shape (N,) experimental delta_delta_g values.
    """
    # 1. Pearson and Spearman Correlations
    mask = ~np.isnan(pred_critical_lams) & ~np.isnan(true_ddgs)
    if np.sum(mask) >= 2:
        pearson_val, _ = pearsonr(pred_critical_lams[mask], true_ddgs[mask])
        spearman_val, _ = spearmanr(pred_critical_lams[mask], true_ddgs[mask])
        # Handle NaN correlation values gracefully if they occur
        if np.isnan(pearson_val): pearson_val = 0.0
        if np.isnan(spearman_val): spearman_val = 0.0
    else:
        pearson_val, spearman_val = 0.0, 0.0
        
    # 2. AUROC (Fold Reversal Classification)
    # Binary target: True delta_delta_g > 0 indicates reversal/dominance swap
    binary_labels = (true_ddgs > 0).astype(int)
    
    if len(np.unique(binary_labels)) > 1:
        auroc_val = roc_auc_score(binary_labels, pred_stability_diffs)
        if np.isnan(auroc_val): auroc_val = 0.5
    else:
        auroc_val = 0.5 # Default baseline for non-diverse labels
        
    return {
        "pearson_corr": pearson_val,
        "spearman_corr": spearman_val,
        "auroc_fold_reversal": auroc_val
    }

def log_pre_registration():
    """
    Hardcoded Pre-registration logging function.
    Prints the evaluation baselines, metrics, and success/null criteria before training starts.
    """
    print("================================================================================")
    print("                            PRE-REGISTRATION LOG                                ")
    print("================================================================================")
    print("Evaluating Model: Implicit Stability Spectroscopy (ISS)")
    print("Date & Time: 2026-06-12 (Locked prior to data evaluation)")
    print("\n[Baseline Models & Targets]")
    print("  - FoldX:          Spearman r = 0.30, AUROC = 0.65")
    print("  - Rosetta ddG:    Spearman r = 0.35, AUROC = 0.70")
    print("\n[Primary Evaluation Metrics]")
    print("  - Spearman rank correlation (predicted critical lambda* vs experimental ddG transition)")
    print("  - AUROC (predicted stability difference m2 - m1 at lambda=0 vs binary ddG sign)")
    print("\n[Null/Success Threshold Criteria]")
    print("  - Success Criteria: Model beats FoldX baseline (Spearman r >= 0.30)")
    print("  - Null Result Criteria: Spearman r < 0.15 or near-zero correlation.")
    print("  - Action on Null Result: Raise 'Null Result: 마진이 \\Delta\\Delta G를 추적하지 못함' and terminate.")
    print("================================================================================")

