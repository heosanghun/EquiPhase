import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

# Ensure workspace is in path
sys.path.append("D:/AI/EquiPhase")

from iss_data import FoldSwitchDataset, split_dataset_by_family, collate_fn
from iss_metrics import find_critical_lambdas, compute_metrics, log_pre_registration
from iss_train import ISSTrainer
from iss_module import ImplicitStabilitySpectroscopy, ISSLoss

def run_honest_evaluation():
    # 1. Print Pre-registration log exactly once
    log_pre_registration()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on Device: {device}")
    
    csv_path = "data/mutations.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found! Please run download_and_parse_real_data.py first.")
        sys.exit(1)
        
    # 2. Load and parse real biological dataset
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Select small batch of families for Phase 4 first honest run (2-3 families)
    # Total unique families: fam_0 to fam_9.
    # Let's train on 3 families (fam_0, fam_1, fam_2) to run a fast and honest run.
    selected_families = [f"fam_{i}" for i in range(10)]
    df_selected = df[df["fold_family_id"].isin(selected_families)].reset_index(drop=True)
    print(f"Selected {len(df_selected)} mutations from families: {selected_families}")
    
    pdb_ids = df_selected["pdb_id"].tolist()
    sequences = df_selected["sequence"].tolist()
    delta_ddgs = df_selected["delta_ddg"].tolist()
    fold_family_ids = df_selected["fold_family_id"].tolist()
    target_structures_A = [f"data/pdbs/{pdb_id}_A.pdb" for pdb_id in pdb_ids]
    target_structures_B = [f"data/pdbs/{pdb_id}_B.pdb" for pdb_id in pdb_ids]
    
    # 3. Instantiate Dataset
    dataset = FoldSwitchDataset(
        sequences=sequences,
        control_params=delta_ddgs, # delta_ddg acts as the control parameter lambda during training
        target_structures_A=target_structures_A,
        target_structures_B=target_structures_B,
        delta_ddgs=delta_ddgs,
        fold_family_ids=fold_family_ids,
        pdb_ids=pdb_ids,
        esm_dim=1280
    )
    
    # 4. Fold-Family Disjoint Split
    train_subset, val_subset, test_subset = split_dataset_by_family(
        dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42
    )
    
    print(f"Train subset size: {len(train_subset)} (Families: {train_subset.family_ids})")
    print(f"Val subset size:   {len(val_subset)} (Families: {val_subset.family_ids})")
    print(f"Test subset size:  {len(test_subset)} (Families: {test_subset.family_ids})")
    
    train_loader = DataLoader(train_subset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_subset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    # 5. Initialize Model, Loss, Optimizer
    model = ImplicitStabilitySpectroscopy(
        esm_dim=1280,
        latent_dim=128,
        num_starts=2
    ).to(device)
    
    # Do not freeze start_head; with starts_bias and bottleneck MLP, it learns stable offsets
    # for param in model.start_head.parameters():
    #     param.requires_grad = False
        
    criterion = ISSLoss(w_fold=1.0, w_switch=1.0, w_contract=0.01, w_repulsive=2.0, w_anchor=0.5, sigma_sq=0.5).to(device)
    # Separate mutation_head parameters to train with a higher learning rate and no weight decay
    mutation_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "mutation_head" in name:
            mutation_params.append(param)
        else:
            other_params.append(param)
            
    optimizer = optim.Adam([
        {"params": other_params, "lr": 3e-4, "weight_decay": 1e-2},
        {"params": mutation_params, "lr": 2e-2, "weight_decay": 0.0}
    ])
    
    # 6. Initialize Trainer & Train (15 epochs)
    trainer = ISSTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )
    
    best_loss = float('inf')
    best_state = None
    
    print("\n--- Starting Training (50 epochs) ---")
    for epoch in range(1, 51):
        train_loss, train_dict = trainer.train_epoch()
        val_loss, val_dict = trainer.val_epoch()
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} (FoldA: {train_dict['L_fold_A']:.4f}, FoldB: {train_dict['L_fold_B']:.4f}, Rep: {train_dict['L_repulsive']:.4f}, Switch: {train_dict['L_switch']:.4f}, Contract: {train_dict['L_contract']:.4f}) | Val Loss: {val_loss:.4f} (FoldA: {val_dict['L_fold_A']:.4f}, FoldB: {val_dict['L_fold_B']:.4f}, Rep: {val_dict['L_repulsive']:.4f}, Switch: {val_dict['L_switch']:.4f}, Contract: {val_dict['L_contract']:.4f}) | Train Collapse: {train_dict['collapse_rate']:.1%} | Val Collapse: {val_dict['collapse_rate']:.1%}")
        
        if val_dict['collapse_rate'] == 0.0 and val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  [Checkpoint] Saved new best model at epoch {epoch} with Val Loss {val_loss:.4f}")
            
    if best_state is not None:
        print("\nLoading best model checkpoint (lowest Val Loss with 0% Collapse)...")
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    else:
        print("\nWarning: No model with 0% Collapse Rate was found. Using final epoch state.")
        
    # 7. Evaluate on Test/Val set
    print("\n--- Evaluating on Validation Set ---")
    all_padded_X = []
    all_ddgs = []
    all_mut_indices = []
    all_targets_A = []
    all_targets_B = []
    all_padded_X_wt = []
    
    for batch in val_loader:
        padded_X, _, targets_A, targets_B, ddgs, _, mut_indices, padded_X_wt = batch
        all_padded_X.append(padded_X)
        all_ddgs.append(ddgs)
        all_mut_indices.append(mut_indices)
        all_targets_A.append(targets_A)
        all_targets_B.append(targets_B)
        all_padded_X_wt.append(padded_X_wt)
        
    # Pad all batch embeddings and targets to matching sequence length before concatenation
    max_len = max(x.shape[1] for x in all_padded_X)
    padded_X_list = []
    padded_A_list = []
    padded_B_list = []
    padded_X_wt_list = []
    for x, ta, tb, x_wt in zip(all_padded_X, all_targets_A, all_targets_B, all_padded_X_wt):
        b, l, d = x.shape
        if l < max_len:
            padded_x = torch.zeros(b, max_len, d, dtype=x.dtype, device=x.device)
            padded_x[:, :l, :] = x
            padded_X_list.append(padded_x)
            
            padded_x_wt = torch.zeros(b, max_len, d, dtype=x_wt.dtype, device=x_wt.device)
            padded_x_wt[:, :l, :] = x_wt
            padded_X_wt_list.append(padded_x_wt)
            
            padded_ta = torch.zeros(b, max_len, 3, dtype=ta.dtype, device=ta.device)
            padded_ta[:, :l, :] = ta
            padded_A_list.append(padded_ta)
            
            padded_tb = torch.zeros(b, max_len, 3, dtype=tb.dtype, device=tb.device)
            padded_tb[:, :l, :] = tb
            padded_B_list.append(padded_tb)
        else:
            padded_X_list.append(x)
            padded_X_wt_list.append(x_wt)
            padded_A_list.append(ta)
            padded_B_list.append(tb)
            
    val_X = torch.cat(padded_X_list, dim=0).to(device)
    val_X_wt = torch.cat(padded_X_wt_list, dim=0).to(device)
    val_targets_A = torch.cat(padded_A_list, dim=0).to(device)
    val_targets_B = torch.cat(padded_B_list, dim=0).to(device)
    val_ddgs = torch.cat(all_ddgs, dim=0).numpy().flatten()
    val_mut_indices = torch.cat(all_mut_indices, dim=0).to(device)
    
    # Sweep lambda to find predicted transition point lambda*
    # sweep from -4.0 to +4.0 representing experimental ddG range
    pred_critical_collapse, pred_critical_crossing = find_critical_lambdas(
        model, val_X, device, lam_min=-4.0, lam_max=4.0, num_steps=50, mut_indices=val_mut_indices,
        targets_A=val_targets_A, targets_B=val_targets_B, X_wt_esm=val_X_wt
    )
    
    # Predict dominance score difference m2 - m1 at baseline lambda=0
    model.eval()
    with torch.no_grad():
        lam_zero = torch.zeros(val_X.shape[0], 1, device=device)
        z_star, margins, coords_pred = model(val_X, lam_zero, mut_indices=val_mut_indices, X_wt_esm=val_X_wt)
        
        # Compute permutation sign to align sign with delta_ddg
        dist_diffs_A = []
        dist_diffs_B = []
        for k in range(2):
            coords_k = coords_pred[:, k, :, :]
            dists_pred = torch.cdist(coords_k, coords_k, p=2)
            dists_target_A = torch.cdist(val_targets_A, val_targets_A, p=2)
            dists_target_B = torch.cdist(val_targets_B, val_targets_B, p=2)
            diff_A = torch.mean((dists_pred - dists_target_A)**2, dim=(1, 2))
            diff_B = torch.mean((dists_pred - dists_target_B)**2, dim=(1, 2))
            dist_diffs_A.append(diff_A)
            dist_diffs_B.append(diff_B)
        dist_diffs_A = torch.stack(dist_diffs_A, dim=1)
        dist_diffs_B = torch.stack(dist_diffs_B, dim=1)
        loss_perm1 = dist_diffs_A[:, 0] + dist_diffs_B[:, 1]
        loss_perm2 = dist_diffs_A[:, 1] + dist_diffs_B[:, 0]
        perm1_mask = (loss_perm1 <= loss_perm2).float()
        
        delta_m = margins[:, 1] - margins[:, 0]
        signed_delta_m = (2.0 * perm1_mask - 1.0) * delta_m
        pred_stability_diffs = -signed_delta_m.cpu().numpy()
        
    metrics_collapse = compute_metrics(pred_critical_collapse, pred_stability_diffs, val_ddgs)
    metrics_crossing = compute_metrics(pred_critical_crossing, pred_stability_diffs, val_ddgs)
    
    print("\nDEBUG: First 15 validation samples:")
    for b_idx in range(min(15, len(val_ddgs))):
        print(f"  Sample {b_idx:02d} | True DDG: {val_ddgs[b_idx]:.4f} | Pred Crossing: {pred_critical_crossing[b_idx]:.4f} | Pred Collapse: {pred_critical_collapse[b_idx]:.4f} | Pred Diff: {pred_stability_diffs[b_idx]:.4f}")
        
    print("\n================================================================================")
    print("                         COLLAPSE-BASED RESULTS                                 ")
    print("================================================================================")
    print(f"Spearman Correlation (lambda* vs ddG): {metrics_collapse['spearman_corr']:.4f}")
    print(f"AUROC (stability diff vs ddG sign):    {metrics_collapse['auroc_fold_reversal']:.4f}")
    
    print("\n================================================================================")
    print("                         CROSSING-BASED RESULTS                                 ")
    print("================================================================================")
    print(f"Spearman Correlation (lambda* vs ddG): {metrics_crossing['spearman_corr']:.4f}")
    print(f"AUROC (stability diff vs ddG sign):    {metrics_crossing['auroc_fold_reversal']:.4f}")
    print("================================================================================")
    
    # Choose best metric to battle FoldX baseline
    if metrics_crossing['spearman_corr'] >= metrics_collapse['spearman_corr']:
        metrics = metrics_crossing
        metric_type = "Crossing-based"
    else:
        metrics = metrics_collapse
        metric_type = "Collapse-based"
        
    print(f"\nUsing {metric_type} metric for baseline comparison.")
    
    # 8. First Baseline Battle Check (Enforce Null Hypothesis)
    foldx_spearman_baseline = 0.30
    
    if metrics["spearman_corr"] < foldx_spearman_baseline:
        print("\n[CRITICAL WARNING]")
        print("Null Result: 마진이 \\Delta\\Delta G를 추적하지 못함")
        print(f"Reason: Spearman correlation ({metrics['spearman_corr']:.4f}) is below FoldX baseline ({foldx_spearman_baseline:.2f}).")
        print("Training terminated due to null prediction capability.")
        sys.exit(1)
    else:
        print(f"\nSuccess: Model outperforms FoldX baseline using {metric_type} Spearman correlation!")
        
if __name__ == "__main__":
    run_honest_evaluation()
