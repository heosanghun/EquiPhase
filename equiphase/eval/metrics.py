import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import hashlib

class RecyclingFabricationException(Exception):
    pass

# Registry to track metric hashes and prevent recycling/duplication (R3)
_run_traces_registry = {}

def check_and_register_run(run_id, config_hash, metrics_dict, training_loss_trace=None):
    """
    Checks if the current run's metrics or training trace are identical to any other registered run.
    Raises RecyclingFabricationException if duplicates are found (R3).
    """
    # Create a string representation of metrics and trace
    serialized = f"metrics:{sorted(metrics_dict.items())}"
    if training_loss_trace is not None:
        serialized += f"|trace:{list(training_loss_trace)}"
        
    run_hash = hashlib.md5(serialized.encode('utf-8')).hexdigest()
    
    for existing_run_id, existing_hash in _run_traces_registry.items():
        if existing_hash == run_hash and existing_run_id != run_id:
            raise RecyclingFabricationException(
                f"INTEGRITY ERROR: Run '{run_id}' has identical metrics/trace to existing run '{existing_run_id}'. "
                f"Recycling or copy-paste detected! Halting execution."
            )
            
    _run_traces_registry[run_id] = run_hash
    print(f"Run '{run_id}' registered successfully (R3 check passed).")

def compute_auprc(y_true, y_pred):
    """Computes Area Under the Precision-Recall Curve."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)

def compute_auroc(y_true, y_pred):
    """Computes Area Under the ROC Curve."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_pred)

def compute_metrics(y_true, y_pred):
    """Computes basic classification metrics."""
    return {
        "AUPRC": float(compute_auprc(y_true, y_pred)),
        "AUROC": float(compute_auroc(y_true, y_pred))
    }

def cluster_block_bootstrap_ci(df, n_iterations=1000, confidence_level=0.95):
    """
    Computes confidence intervals using cluster block bootstrap (resampling families/clusters, not rows).
    df: DataFrame containing ['cluster_id', 'label', 'pred']
    """
    unique_clusters = df['cluster_id'].unique()
    n_clusters = len(unique_clusters)
    
    boot_auprcs = []
    boot_aurocs = []
    
    np.random.seed(42)  # For reproducibility
    
    # Pre-group data by cluster_id to speed up resampling
    grouped = {cid: group[['label', 'pred']].values for cid, group in df.groupby('cluster_id')}
    
    for _ in range(n_iterations):
        boot_cids = np.random.choice(unique_clusters, size=n_clusters, replace=True)
        
        # Aggregate values for bootstrapped clusters
        boot_y_true = []
        boot_y_pred = []
        for cid in boot_cids:
            vals = grouped[cid]
            boot_y_true.append(vals[:, 0])
            boot_y_pred.append(vals[:, 1])
            
        boot_y_true = np.concatenate(boot_y_true)
        boot_y_pred = np.concatenate(boot_y_pred)
        
        boot_auprcs.append(compute_auprc(boot_y_true, boot_y_pred))
        boot_aurocs.append(compute_auroc(boot_y_true, boot_y_pred))
        
    alpha = 1.0 - confidence_level
    lower_pct = 100 * (alpha / 2.0)
    upper_pct = 100 * (1.0 - alpha / 2.0)
    
    return {
        "AUPRC_mean": float(np.mean(boot_auprcs)),
        "AUPRC_median": float(np.median(boot_auprcs)),
        "AUPRC_ci": [float(np.percentile(boot_auprcs, lower_pct)), float(np.percentile(boot_auprcs, upper_pct))],
        "AUROC_mean": float(np.mean(boot_aurocs)),
        "AUROC_median": float(np.median(boot_aurocs)),
        "AUROC_ci": [float(np.percentile(boot_aurocs, lower_pct)), float(np.percentile(boot_aurocs, upper_pct))]
    }
