import numpy as np
import torch
from scipy.stats import pearsonr

def compute_partial_correlation(x, y, z):
    """
    Computes the partial correlation of x and y, controlling for z.
    x, y, z must be 1D numpy arrays of the same size.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    # Linear regression of x on z: x = a_x * z + b_x
    slope_x, intercept_x = np.polyfit(z, x, 1)
    res_x = x - (slope_x * z + intercept_x)
    
    # Linear regression of y on z: y = a_y * z + b_y
    slope_y, intercept_y = np.polyfit(z, y, 1)
    res_y = y - (slope_y * z + intercept_y)
    
    # Pearson correlation coefficient between the residuals
    r_val, p_val = pearsonr(res_x, res_y)
    return r_val, p_val, res_x, res_y

def compute_auroc(y_true, y_scores):
    """
    Computes Area Under the ROC Curve.
    """
    if hasattr(y_true, 'cpu'):
        y_true = y_true.cpu().numpy()
    if hasattr(y_scores, 'cpu'):
        y_scores = y_scores.cpu().numpy()
        
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, y_scores)
    except ImportError:
        # Fallback manual AUROC
        desc_score_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[desc_score_indices]
        num_pos = np.sum(y_true == 1)
        num_neg = np.sum(y_true == 0)
        if num_pos == 0 or num_neg == 0:
            return 0.5
        tps = np.cumsum(y_true_sorted == 1)
        fps = np.cumsum(y_true_sorted == 0)
        tpr = tps / num_pos
        fpr = fps / num_neg
        auc = 0.0
        for i in range(1, len(fpr)):
            auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2.0
        auc += fpr[0] * tpr[0] / 2.0
        return auc

def run_label_permutation_audit(model, val_loader, device):
    """
    Permutes the validation labels, evaluates the model predictions,
    and returns the resulting AUROC (which should regress to ~0.5).
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            padded_X, lams, targets_A, targets_B, ddgs, _, mut_indices, padded_X_wt = batch
            
            padded_X = padded_X.to(device)
            lams = lams.to(device)
            mut_indices = mut_indices.to(device)
            padded_X_wt = padded_X_wt.to(device)
            
            # Predict stability margins
            _, margins, _ = model(padded_X, lams, mut_indices=mut_indices, X_wt_esm=padded_X_wt)
            
            # Use margin difference as prediction score for bistability transition classification
            # margins[:, 1] is Fold B, margins[:, 0] is Fold A.
            pred_score = (margins[:, 1] - margins[:, 0]).cpu().numpy()
            all_preds.extend(pred_score)
            
            # Binary classification target based on control parameter lambda (transition occurs at lambda > 0.5)
            binary_target = (lams.squeeze(-1) > 0.5).long().cpu().numpy()
            all_targets.extend(binary_target)
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Permute targets
    perm_indices = torch.randperm(len(all_targets)).numpy()
    permuted_targets = all_targets[perm_indices]
    
    original_auroc = compute_auroc(all_targets, all_preds)
    permuted_auroc = compute_auroc(permuted_targets, all_preds)
    
    return original_auroc, permuted_auroc
