import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import random
import time
import json
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

# Ensure workspace is in path
sys.path.append("D:/AI/EquiPhase")

from iss_data import parse_pdb, get_esm2_embeddings, collate_fn
from iss_module import ImplicitStabilitySpectroscopy, ISSLoss
from iss_train import ISSTrainer

def kabsch_rmsd(P, Q):
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    H = np.dot(P_centered.T, Q_centered)
    U, S, Vt = np.linalg.svd(H)
    
    d = np.linalg.det(np.dot(Vt.T, U.T))
    F = np.eye(3)
    if d < 0.0:
        F[2, 2] = -1.0
    R = np.dot(Vt.T, np.dot(F, U.T))
    
    P_rotated = np.dot(P_centered, R)
    diff = P_rotated - Q_centered
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return rmsd

def generate_low_freq_decoy(coords, target_rmsd):
    # coords: numpy array of shape (L, 3)
    L = coords.shape[0]
    
    # 1. Generate random walk
    perturb = np.zeros((L, 3))
    for i in range(1, L):
        perturb[i] = perturb[i-1] + np.random.randn(3)
        
    # 2. Smooth the random walk using a moving average (low-frequency smoothing)
    window_size = min(15, L)
    smoothed = np.zeros((L, 3))
    for i in range(L):
        start = max(0, i - window_size // 2)
        end = min(L, i + window_size // 2 + 1)
        smoothed[i] = np.mean(perturb[start:end], axis=0)
        
    # Center the perturbation to prevent translation shifts
    smoothed = smoothed - np.mean(smoothed, axis=0)
    
    def get_aligned_rmsd(s):
        decoy_candidate = coords + s * smoothed
        return kabsch_rmsd(coords, decoy_candidate)
        
    # Binary search for scale s
    low, high = 0.0, 100.0
    for _ in range(30):
        mid = (low + high) / 2
        r = get_aligned_rmsd(mid)
        if r < target_rmsd:
            low = mid
        else:
            high = mid
            
    s_final = (low + high) / 2
    decoy = coords + s_final * smoothed
    return decoy

def optimize_latent_state(model, X_esm, Y_coords, num_steps=100, lr=0.1):
    device = X_esm.device
    L = X_esm.shape[0]
    
    with torch.no_grad():
        X_proj = model.esm_proj(X_esm.unsqueeze(0)).detach()
        X_pooled = torch.mean(X_proj, dim=1).detach()
        
    z = torch.zeros(1, model.latent_dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=lr)
    
    Y_target = torch.tensor(Y_coords, dtype=torch.float32, device=device)
    
    for _ in range(num_steps):
        optimizer.zero_grad()
        z_rep = z.unsqueeze(1).repeat(1, L, 1)
        z_mixed = model.mix_layer(torch.cat([z_rep, X_proj], dim=-1))
        coords_pred = model.coord_head(z_mixed).squeeze(0)
        
        loss = torch.mean((coords_pred - Y_target)**2)
        loss.backward()
        optimizer.step()
        
    return z.detach(), X_pooled.detach()

class Phase2Dataset(Dataset):
    def __init__(self, df_pairs, cached_embs):
        self.df_pairs = df_pairs
        self.cached_embs = cached_embs
        
    def __len__(self):
        return len(self.df_pairs)
        
    def __getitem__(self, idx):
        row = self.df_pairs.iloc[idx]
        p1 = row['pdb1']
        family_id = row['family_id']
        X_esm = self.cached_embs[p1]
        L = X_esm.shape[0]
        
        # Unsupervised training dummy variables
        lam = torch.tensor(0.0)
        target_A = torch.zeros(L, 3)
        target_B = torch.zeros(L, 3)
        ddg = torch.tensor(0.0)
        mut_idx = -1
        X_wt_esm = X_esm
        
        return X_esm, lam, target_A, target_B, ddg, family_id, mut_idx, X_wt_esm

def bootstrap_auroc(y_true, y_pred, num_boots=1000):
    n = len(y_true)
    boot_stats = []
    for _ in range(num_boots):
        indices = np.random.choice(n, size=n, replace=True)
        if len(np.unique(y_true[indices])) == 2:
            boot_stats.append(roc_auc_score(y_true[indices], y_pred[indices]))
    if not boot_stats:
        return 0.5, (0.5, 0.5)
    boot_stats = sorted(boot_stats)
    lower = np.percentile(boot_stats, 2.5)
    upper = np.percentile(boot_stats, 97.5)
    mean = np.mean(boot_stats)
    return mean, (lower, upper)

def bootstrap_auroc_diff(y_true, y_pred1, y_pred2, num_boots=1000):
    n = len(y_true)
    boot_stats = []
    for _ in range(num_boots):
        indices = np.random.choice(n, size=n, replace=True)
        if len(np.unique(y_true[indices])) == 2:
            auc1 = roc_auc_score(y_true[indices], y_pred1[indices])
            auc2 = roc_auc_score(y_true[indices], y_pred2[indices])
            boot_stats.append(auc1 - auc2)
    if not boot_stats:
        return 0.0, (0.0, 0.0)
    boot_stats = sorted(boot_stats)
    lower = np.percentile(boot_stats, 2.5)
    upper = np.percentile(boot_stats, 97.5)
    mean = np.mean(boot_stats)
    return mean, (lower, upper)

def main():
    print("==========================================================")
    print("      ISS Phase 2: Real-Data Detector Hypothesis          ")
    print("==========================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    df_pairs = pd.read_csv("data/benchmark_pairs.csv")
    df_plddt = pd.read_csv("data/imputed_benchmark_plddt.csv")
    
    assert (df_pairs['pdb1'] == df_plddt['pdb1']).all(), "Alignment mismatch on pdb1!"
    assert (df_pairs['pdb2'] == df_plddt['pdb2']).all(), "Alignment mismatch on pdb2!"
    assert (df_pairs['is_switcher'] == df_plddt['is_switcher']).all(), "Alignment mismatch on labels!"
    
    # 1. Parse and extract all sequences & verify PDB files (STEP 1 - Block A)
    print("\n--- STEP 1: Real Data Pipeline Audit (Block A) ---")
    unique_pdbs = set(df_pairs['pdb1'].tolist() + df_pairs['pdb2'].tolist())
    print(f"Unique PDB chains: {len(unique_pdbs)}")
    
    parsed_seqs = {}
    parsed_coords = {}
    
    for pdb in unique_pdbs:
        path = f"data/clean_chains/{pdb}.pdb"
        if not os.path.exists(path):
            print(f"Block A FAILED: Missing structure file {path}")
            sys.exit(1)
        seq, coords = parse_pdb(path)
        if seq is None or coords is None:
            print(f"Block A FAILED: Could not parse structure {path}")
            sys.exit(1)
        parsed_seqs[pdb] = seq
        parsed_coords[pdb] = coords
        
    print(f"Successfully parsed all {len(unique_pdbs)} clean PDB files. Model 1 only.")
    
    # Pre-compute ESM-2 embeddings once to speed up training
    print("Pre-computing ESM-2 embeddings...")
    pdb_list = list(unique_pdbs)
    seq_list = [parsed_seqs[pdb] for pdb in pdb_list]
    esm_embeddings = get_esm2_embeddings(seq_list)
    cached_embeddings = {pdb_list[i]: esm_embeddings[i] for i in range(len(pdb_list))}
    print("Embeddings cached successfully.")
    
    # Calculate execution integrity hash
    all_seqs_str = "".join(sorted([parsed_seqs[pdb] for pdb in unique_pdbs]))
    import hashlib
    execution_hash = hashlib.sha256(all_seqs_str.encode('utf-8')).hexdigest()
    print(f"Execution Integrity Hash: {execution_hash}")
    
    # 2. Sequence Clustering Proxy (30% global identity)
    chain_list = list(unique_pdbs)
    chain_to_fam = {}
    fam_counter = 0
    visited = set()
    for chain1 in chain_list:
        if chain1 in visited:
            continue
        current_fam = f"fam_{fam_counter}"
        fam_counter += 1
        queue = [chain1]
        visited.add(chain1)
        while queue:
            curr = queue.pop(0)
            chain_to_fam[curr] = current_fam
            seq_curr = parsed_seqs[curr]
            
            for other in chain_list:
                if other in visited:
                    continue
                seq_other = parsed_seqs[other]
                len_ratio = len(seq_curr) / len(seq_other)
                if len_ratio < 0.5 or len_ratio > 2.0:
                    continue
                n1, n2 = len(seq_curr), len(seq_other)
                min_len = min(n1, n2)
                mismatches = sum(1 for a, b in zip(seq_curr[:min_len], seq_other[:min_len]) if a != b)
                identity = (min_len - mismatches) / max(n1, n2)
                if identity >= 0.30:
                    visited.add(other)
                    queue.append(other)
                    
    df_pairs['family_id'] = [chain_to_fam[row['pdb1']] for _, row in df_pairs.iterrows()]
    unique_fams = df_pairs['family_id'].unique().tolist()
    print(f"Grouped {len(df_pairs)} pairs into {len(unique_fams)} disjoint sequence families.")
    
    # Baselines
    pair_rmsds = df_pairs['pair_rmsd'].values
    true_labels = df_pairs['is_switcher'].values
    
    # pLDDT baseline mask: exclude imputed/missing (NaN) values
    plddt_raw1 = df_plddt['plddt1'].values
    plddt_raw2 = df_plddt['plddt2'].values
    valid_plddt_mask = ~np.isnan(plddt_raw1) & ~np.isnan(plddt_raw2)
    plddt_scores = -0.5 * (plddt_raw1 + plddt_raw2)
    
    print(f"pLDDT baseline data: {valid_plddt_mask.sum()} valid pairs, {len(df_pairs) - valid_plddt_mask.sum()} missing pairs excluded.")
    
    # 5-seed evaluation
    seeds = [42, 100, 2026, 777, 999]
    seed_results = []
    
    for seed in seeds:
        print(f"\n==================== Running Evaluation on Seed {seed} ====================")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        shuffled_fams = unique_fams.copy()
        random.shuffle(shuffled_fams)
        
        oof_min_margins = np.zeros(len(df_pairs))      # min(mA, mB)
        oof_neg_min_margins = np.zeros(len(df_pairs))  # -min(mA, mB)
        oof_decoy_margins = np.zeros(len(df_pairs))    # -min(mA, mDecoy)
        
        for fold in range(5):
            val_fams = set(shuffled_fams[fold::5])
            
            train_idx = df_pairs[~df_pairs['family_id'].isin(val_fams)].index.tolist()
            val_idx = df_pairs[df_pairs['family_id'].isin(val_fams)].index.tolist()
            
            train_sub = df_pairs.iloc[train_idx]
            val_sub = df_pairs.iloc[val_idx]
            
            train_ds = Phase2Dataset(train_sub, cached_embeddings)
            val_ds = Phase2Dataset(val_sub, cached_embeddings)
            
            train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
            
            model = ImplicitStabilitySpectroscopy(esm_dim=1280, latent_dim=64, num_starts=2).to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            
            criterion = ISSLoss(
                w_switch=0.0,
                w_gnm=1.0,
                w_contact=1.0,
                w_phys=2.0,
                w_repulsive=5.0,
                w_contract=0.01,
                w_anchor=0.5
            ).to(device)
            
            trainer = ISSTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device
            )
            
            trainer.fit(epochs=10)
            
            model.eval()
            for idx in val_idx:
                row = df_pairs.iloc[idx]
                p1, p2 = row['pdb1'], row['pdb2']
                is_switch = row['is_switcher']
                rmsd_val = row['pair_rmsd']
                
                coords1 = parsed_coords[p1]
                coords2 = parsed_coords[p2]
                
                emb1 = cached_embeddings[p1].to(device)
                emb2 = cached_embeddings[p2].to(device)
                
                z_A, X_pooled_A = optimize_latent_state(model, emb1, coords1)
                z_B, X_pooled_B = optimize_latent_state(model, emb2, coords2)
                
                mA = model.compute_stability_margin(z_A, X_pooled_A, torch.zeros(1, 1, device=device), X_pooled_A).item()
                mB = model.compute_stability_margin(z_B, X_pooled_B, torch.zeros(1, 1, device=device), X_pooled_B).item()
                
                oof_min_margins[idx] = min(mA, mB)
                oof_neg_min_margins[idx] = -min(mA, mB)
                
                # B2 Decoy Conformation Control
                if is_switch == 1:
                    decoy_coords = generate_low_freq_decoy(coords1, rmsd_val)
                    z_decoy, X_pooled_decoy = optimize_latent_state(model, emb1, decoy_coords)
                    mDecoy = model.compute_stability_margin(z_decoy, X_pooled_decoy, torch.zeros(1, 1, device=device), X_pooled_decoy).item()
                    oof_decoy_margins[idx] = -min(mA, mDecoy)
                else:
                    oof_decoy_margins[idx] = -min(mA, mB)
                    
        # Seed metrics calculation
        # 1. Report both directions:
        auc_margin_neg, ci_margin_neg = bootstrap_auroc(true_labels, oof_neg_min_margins)
        auc_margin_pos, ci_margin_pos = bootstrap_auroc(true_labels, oof_min_margins)
        
        # We lock the primary margin predictions to the hypothesized negative sign direction:
        auc_margin = auc_margin_neg
        ci_margin = ci_margin_neg
        
        auc_rmsd, ci_rmsd = bootstrap_auroc(true_labels, pair_rmsds)
        
        # Calculate pLDDT baseline AUROC on the non-imputed subset only
        auc_plddt, ci_plddt = bootstrap_auroc(true_labels[valid_plddt_mask], plddt_scores[valid_plddt_mask])
        
        auc_diff_rmsd, ci_diff_rmsd = bootstrap_auroc_diff(true_labels, oof_neg_min_margins, pair_rmsds)
        
        # B1 Control: Label Shuffle
        shuffled_labels = true_labels.copy()
        random.shuffle(shuffled_labels)
        auc_shuffle, _ = bootstrap_auroc(shuffled_labels, oof_neg_min_margins)
        
        # B2 Control: Distinguishability between real switcher fold B and decoy
        # We classify real switchers (labels=1) vs decoy switchers (labels=0)
        switcher_indices = df_pairs[df_pairs['is_switcher'] == 1].index.tolist()
        y_b2 = np.ones(len(switcher_indices) * 2)
        y_b2[len(switcher_indices):] = 0.0
        scores_b2 = np.concatenate([oof_neg_min_margins[switcher_indices], oof_decoy_margins[switcher_indices]])
        auc_decoy_dist, _ = bootstrap_auroc(y_b2, scores_b2)
        
        # Fit logistic regression for partial correlation
        rmsd_std = (pair_rmsds - pair_rmsds.mean()) / pair_rmsds.std()
        margin_std = (oof_neg_min_margins - oof_neg_min_margins.mean()) / oof_neg_min_margins.std()
        X_lr = pd.DataFrame({
            'intercept': 1.0,
            'rmsd': rmsd_std,
            'margin': margin_std
        })
        lr_model = sm.Logit(true_labels, X_lr).fit(disp=0)
        p_val_margin = lr_model.pvalues['margin']
        coef_margin = lr_model.params['margin']
        
        # Dist by class
        margin_switcher = oof_neg_min_margins[true_labels == 1]
        margin_control = oof_neg_min_margins[true_labels == 0]
        
        print(f"\nSeed {seed} Results:")
        print(f"  Stability Margin AUROC (neg direction): {auc_margin_neg:.4f} (95% CI: {ci_margin_neg[0]:.4f} - {ci_margin_neg[1]:.4f})")
        print(f"  Stability Margin AUROC (pos direction): {auc_margin_pos:.4f} (95% CI: {ci_margin_pos[0]:.4f} - {ci_margin_pos[1]:.4f})")
        print(f"  pair_rmsd Baseline AUROC:             {auc_rmsd:.4f} (95% CI: {ci_rmsd[0]:.4f} - {ci_rmsd[1]:.4f})")
        print(f"  pLDDT Baseline AUROC (excluding missing): {auc_plddt:.4f} (95% CI: {ci_plddt[0]:.4f} - {ci_plddt[1]:.4f})")
        print(f"  AUROC Diff (Margin - RMSD):            {auc_diff_rmsd:.4f} (95% CI: {ci_diff_rmsd[0]:.4f} - {ci_diff_rmsd[1]:.4f})")
        print(f"  B1 (Label Shuffle) AUROC:             {auc_shuffle:.4f}")
        print(f"  B2 (Real vs Decoy) AUROC:             {auc_decoy_dist:.4f}")
        print(f"  Logistic Reg Margin Coef:             {coef_margin:.4f} (p-value: {p_val_margin:.4e})")
        
        seed_results.append({
            "seed": seed,
            "auc_margin_neg": auc_margin_neg, "ci_margin_neg": ci_margin_neg,
            "auc_margin_pos": auc_margin_pos, "ci_margin_pos": ci_margin_pos,
            "auc_rmsd": auc_rmsd, "ci_rmsd": ci_rmsd,
            "auc_plddt": auc_plddt, "ci_plddt": ci_plddt,
            "auc_diff_rmsd": auc_diff_rmsd, "ci_diff_rmsd": ci_diff_rmsd,
            "auc_shuffle": auc_shuffle,
            "auc_decoy_dist": auc_decoy_dist,
            "coef_margin": coef_margin,
            "p_val_margin": p_val_margin,
            "margin_switcher_mean": margin_switcher.mean(),
            "margin_switcher_std": margin_switcher.std(),
            "margin_control_mean": margin_control.mean(),
            "margin_control_std": margin_control.std()
        })
        
    # Aggregate results across all seeds
    mean_margin_neg = np.mean([r['auc_margin_neg'] for r in seed_results])
    mean_margin_pos = np.mean([r['auc_margin_pos'] for r in seed_results])
    mean_rmsd = np.mean([r['auc_rmsd'] for r in seed_results])
    mean_plddt = np.mean([r['auc_plddt'] for r in seed_results])
    mean_diff_rmsd = np.mean([r['auc_diff_rmsd'] for r in seed_results])
    mean_shuffle = np.mean([r['auc_shuffle'] for r in seed_results])
    mean_decoy_dist = np.mean([r['auc_decoy_dist'] for r in seed_results])
    mean_p_val = np.mean([r['p_val_margin'] for r in seed_results])
    
    ci_diff_lows = [r['ci_diff_rmsd'][0] for r in seed_results]
    p_vals = [r['p_val_margin'] for r in seed_results]
    
    print("\n" + "="*80)
    print("                      PHASE 2 DETECTOR HARNESS SUMMARY                          ")
    print("="*80)
    print(f"Average Stability Margin AUROC (neg direction): {mean_margin_neg:.4f}")
    print(f"Average Stability Margin AUROC (pos direction): {mean_margin_pos:.4f}")
    print(f"Average pair_rmsd Baseline AUROC:             {mean_rmsd:.4f}")
    print(f"Average pLDDT Baseline AUROC (actual only):   {mean_plddt:.4f}")
    print(f"Average AUROC Diff (Margin-RMSD):             {mean_diff_rmsd:.4f}")
    print(f"Average B1 (Label Shuffle) AUROC:             {mean_shuffle:.4f}")
    print(f"Average B2 (Real vs Decoy) AUROC:             {mean_decoy_dist:.4f}")
    print(f"Average Logistic Reg Margin p-val:            {mean_p_val:.4e}")
    print("-"*80)
    
    # 3. Verdict Determination
    # B1 must destroy predictive power (mean_shuffle ~ 0.50)
    # B2 must show that margin distinguishes real switcher fold from random matched-RMSD decoy
    b1_pass = (mean_shuffle < 0.55)
    b2_pass = (mean_decoy_dist > 0.60)
    
    control_pass = b1_pass and b2_pass
    
    if not control_pass:
        verdict = "NULL"
        reason = f"Controls failed. B1 Pass: {b1_pass} (shuffle={mean_shuffle:.4f}) | B2 Pass: {b2_pass} (decoy_dist={mean_decoy_dist:.4f})"
    else:
        # Check if margin outperforms pair_rmsd significantly (pairwise difference strictly > 0 for all seeds)
        all_seeds_significant_diff = all(low > 0 for low in ci_diff_lows)
        # Check if margin remains significant after controlling for RMSD in logistic regression
        all_seeds_significant_lr = all(p < 0.05 for p in p_vals)
        
        if all_seeds_significant_diff and all_seeds_significant_lr:
            verdict = "PASS"
            reason = "Jacobian stability margin significantly outperforms pair_rmsd baseline and remains significant under partial correlation control across all seeds."
        elif mean_margin_neg > mean_rmsd:
            verdict = "UNDERPOWERED"
            reason = "Jacobian stability margin shows higher point estimate than pair_rmsd, but bootstrap CI overlaps with 0 or logistic significance is inconsistent."
        else:
            verdict = "NULL"
            reason = "Jacobian stability margin does not outperform the pair_rmsd baseline."
            
    print(f"VERDICT: {verdict}")
    print(f"Reason:  {reason}")
    print("="*80)
    
    # Save final report JSON
    report = {
        "integrity_hash": execution_hash,
        "seed_results": seed_results,
        "averages": {
            "margin_neg_auroc": mean_margin_neg,
            "margin_pos_auroc": mean_margin_pos,
            "rmsd_auroc": mean_rmsd,
            "plddt_auroc": mean_plddt,
            "diff_rmsd_auroc": mean_diff_rmsd,
            "shuffle_auroc": mean_shuffle,
            "decoy_dist_auroc": mean_decoy_dist,
            "p_val_margin": mean_p_val
        },
        "verdict": verdict,
        "reason": reason
    }
    with open("data/phase2_verdict_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
if __name__ == "__main__":
    main()
