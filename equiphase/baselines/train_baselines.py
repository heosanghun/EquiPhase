import os
# Set Hugging Face mirror endpoint to bypass LFS CDN blocks
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

# Monkey-patch platform module to bypass Windows WMI query hangs
import platform
from collections import namedtuple
UnameResult = namedtuple('UnameResult', ['system', 'node', 'release', 'version', 'machine', 'processor'])
platform.win32_ver = lambda *args, **kwargs: ('10', '10.0.0', '', 'Multiprocessor Free')
platform.uname = lambda: UnameResult('Windows', 'DESKTOP-XXX', '10', '10.0.0', 'AMD64', 'AMD64')
platform.machine = lambda: 'AMD64'
platform.system = lambda: 'Windows'
platform.processor = lambda: 'AMD64'
platform.release = lambda: '10'
platform.version = lambda: '10.0.0'

import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import sys

BASE_DIR = "D:/AI/EquiPhase/"
DATA_DIR = os.path.join(BASE_DIR, 'equiphase', 'data')
RESULTS_PATH = os.path.join(BASE_DIR, "baselines_results.json")

# Ensure eval is in path to import metrics
sys.path.append(os.path.join(BASE_DIR))
import hashlib
from equiphase.eval.metrics import compute_metrics, cluster_block_bootstrap_ci, check_and_register_run, compute_auprc

# Hyperparameters
EPOCHS = 25
BATCH_SIZE = 64
LR = 1e-3
SEEDS = [42, 100, 2026, 777, 999]

# Models definition
class VanillaMLP(nn.Module):
    def __init__(self, input_dim):
        super(VanillaMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc3(x)).squeeze(-1)

class EnergyBasedMLP(nn.Module):
    def __init__(self, input_dim):
        super(EnergyBasedMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Outputs energy for class 0 and class 1
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        energy = self.fc3(x)  # (B, 2)
        return F.log_softmax(-energy, dim=-1)

# Normalization helper
def normalize_conditions(df):
    log_c = np.log10(df['parsed_c_sat'].values + 1e-3)
    log_c_norm = (log_c - np.mean(log_c)) / (np.std(log_c) + 1e-5)
    
    salt_norm = df['parsed_salt'].values / 1000.0
    ph_norm = df['parsed_ph'].values / 14.0
    temp_norm = df['parsed_temp'].values / 100.0
    
    conds = np.stack([log_c_norm, salt_norm, ph_norm, temp_norm], axis=1)
    return torch.tensor(conds, dtype=torch.float32)

def compute_biophysical_score(seq):
    if not isinstance(seq, str) or len(seq) == 0:
        return 0.0
    promote = sum(1 for c in seq if c in 'RGQNYSF')
    inhibit = sum(1 for c in seq if c == 'P')
    score = (promote - inhibit) / len(seq)
    return 1.0 / (1.0 + np.exp(-10.0 * (score - 0.08)))

def simulate_empirical_power(n_families, n_records, sigma_family, true_delta=0.05, n_trials=1000, n_boot=500):
    np.random.seed(42)
    family_sizes = np.random.poisson(lam=(n_records / n_families - 1), size=n_families) + 1
    diff = n_records - sum(family_sizes)
    for i in range(abs(diff)):
        idx = i % n_families
        if diff > 0:
            family_sizes[idx] += 1
        elif diff < 0 and family_sizes[idx] > 1:
            family_sizes[idx] -= 1
            
    successful_trials = 0
    sigma_record = 0.02
    
    for trial in range(n_trials):
        fam_baseline = np.random.normal(0.70, sigma_family, size=n_families)
        fam_deq = fam_baseline + np.random.normal(true_delta, 0.03, size=n_families)
        
        fam_baseline = np.clip(fam_baseline, 0.4, 0.99)
        fam_deq = np.clip(fam_deq, 0.4, 0.99)
        
        family_scores = {}
        for idx in range(n_families):
            n_recs = family_sizes[idx]
            b_sc = fam_baseline[idx] + np.random.normal(0, sigma_record, size=n_recs)
            d_sc = fam_deq[idx] + np.random.normal(0, sigma_record, size=n_recs)
            family_scores[idx] = (d_sc.sum(), b_sc.sum(), n_recs)
            
        bootstrap_deltas = []
        for boot in range(n_boot):
            boot_families = np.random.choice(range(n_families), size=n_families, replace=True)
            tot_deq = 0.0
            tot_base = 0.0
            tot_n = 0
            for f in boot_families:
                d_sum, b_sum, n_recs = family_scores[f]
                tot_deq += d_sum
                tot_base += b_sum
                tot_n += n_recs
            bootstrap_deltas.append((tot_deq - tot_base) / tot_n)
            
        ci_lower = np.percentile(bootstrap_deltas, 2.5)
        median_delta = np.median(bootstrap_deltas)
        
        if ci_lower > 0 and median_delta >= 0.05:
            successful_trials += 1
            
    power_0_05 = successful_trials / n_trials
    
    # Find minimum detectable effect with >= 80% power
    mde = 0.05
    for test_delta in np.arange(0.05, 0.15, 0.01):
        successful_trials_mde = 0
        for trial in range(500):
            fam_baseline = np.random.normal(0.70, sigma_family, size=n_families)
            fam_deq = fam_baseline + np.random.normal(test_delta, 0.03, size=n_families)
            
            fam_baseline = np.clip(fam_baseline, 0.4, 0.99)
            fam_deq = np.clip(fam_deq, 0.4, 0.99)
            
            family_scores = {}
            for idx in range(n_families):
                n_recs = family_sizes[idx]
                b_sc = fam_baseline[idx] + np.random.normal(0, sigma_record, size=n_recs)
                d_sc = fam_deq[idx] + np.random.normal(0, sigma_record, size=n_recs)
                family_scores[idx] = (d_sc.sum(), b_sc.sum(), n_recs)
                
            bootstrap_deltas = []
            for boot in range(200):
                boot_families = np.random.choice(range(n_families), size=n_families, replace=True)
                tot_deq = 0.0
                tot_base = 0.0
                tot_n = 0
                for f in boot_families:
                    d_sum, b_sum, n_recs = family_scores[f]
                    tot_deq += d_sum
                    tot_base += b_sum
                    tot_n += n_recs
                bootstrap_deltas.append((tot_deq - tot_base) / tot_n)
                
            ci_lower = np.percentile(bootstrap_deltas, 2.5)
            median_delta = np.median(bootstrap_deltas)
            if ci_lower > 0 and median_delta >= 0.05:
                successful_trials_mde += 1
        mde_power = successful_trials_mde / 500
        if mde_power >= 0.80:
            mde = test_delta
            break
            
    return power_0_05, mde

def train_model(model_type, X_train, y_train, X_val, y_val, seed, run_id, config_hash):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == "energy_based":
        model = EnergyBasedMLP(X_train.shape[1]).to(device)
        criterion = nn.NLLLoss()
    else:
        model = VanillaMLP(X_train.shape[1]).to(device)
        criterion = nn.BCELoss()
        
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    loss_trace = []
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            if model_type == "energy_based":
                log_p = model(batch_x)
                loss = criterion(log_p, batch_y.long())
            else:
                pred = model(batch_x)
                loss = criterion(pred, batch_y.float())
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        loss_trace.append(np.mean(epoch_losses))
        
    model.eval()
    val_preds = []
    with torch.no_grad():
        for batch_x, _ in val_loader:
            batch_x = batch_x.to(device)
            if model_type == "energy_based":
                log_p = model(batch_x)
                pred = torch.exp(log_p[:, 1])
            else:
                pred = model(batch_x)
            val_preds.extend(pred.cpu().numpy())
            
    val_preds = np.array(val_preds)
    metrics = compute_metrics(y_val.numpy(), val_preds)
    
    check_and_register_run(run_id, config_hash, metrics, training_loss_trace=loss_trace)
    return val_preds

def train_active_learning(X_train_full, y_train_full, X_val, y_val, seed, run_id, config_hash):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_train = len(X_train_full)
    initial_budget = max(int(0.10 * n_train), 1)
    indices = np.arange(n_train)
    active_indices = list(np.random.choice(indices, size=initial_budget, replace=False))
    remaining_indices = list(set(indices) - set(active_indices))
    
    n_cycles = 4
    add_size = int(0.10 * n_train)
    
    for cycle in range(n_cycles):
        X_train_sub = X_train_full[active_indices]
        y_train_sub = y_train_full[active_indices]
        
        model = VanillaMLP(X_train_full.shape[1]).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        
        dataset = TensorDataset(X_train_sub, y_train_sub)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        for epoch in range(10):
            model.train()
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y.float())
                loss.backward()
                optimizer.step()
                
        model.eval()
        X_pool = X_train_full[remaining_indices]
        with torch.no_grad():
            pool_preds = model(X_pool.to(device)).cpu().numpy()
            
        p = np.clip(pool_preds, 1e-8, 1.0 - 1e-8)
        entropy = -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)
        
        query_idx_sorted = np.argsort(entropy)[::-1]
        top_queries = query_idx_sorted[:add_size]
        
        queried_indices_actual = [remaining_indices[idx] for idx in top_queries]
        active_indices.extend(queried_indices_actual)
        remaining_indices = list(set(remaining_indices) - set(queried_indices_actual))
        
    X_train_active = X_train_full[active_indices]
    y_train_active = y_train_full[active_indices]
    
    return train_model("vanilla", X_train_active, y_train_active, X_val, y_val, seed, run_id, config_hash)

def main():
    print("Loading data splits...")
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train.tsv"), sep="\t")
    df_val = pd.read_csv(os.path.join(DATA_DIR, "val.tsv"), sep="\t")
    
    emb_path = os.path.join(DATA_DIR, "esm2_embeddings.pkl")
    if not os.path.exists(emb_path):
        print(f"ERROR: ESM-2 embeddings cache file not found at {emb_path}. Run precompute script first.")
        return
        
    with open(emb_path, "rb") as f:
        embeddings = pickle.load(f)
        
    print(f"Loaded embeddings for {len(embeddings)} unique sequences.")
    
    sample_seq = list(embeddings.keys())[0]
    emb_dim = len(embeddings[sample_seq])
    print(f"Embedding dimension: {emb_dim}")
    
    train_embs = torch.tensor(np.stack(df_train['Sequence'].map(embeddings).values), dtype=torch.float32)
    val_embs = torch.tensor(np.stack(df_val['Sequence'].map(embeddings).values), dtype=torch.float32)
    
    y_train = torch.tensor(df_train['label'].values, dtype=torch.float32)
    y_val = torch.tensor(df_val['label'].values, dtype=torch.float32)
    
    train_conds = normalize_conditions(df_train)
    val_conds = normalize_conditions(df_val)
    
    X_train_seq = train_embs
    X_val_seq = val_embs
    
    X_train_cond = torch.cat([train_embs, train_conds], dim=1)
    X_val_cond = torch.cat([val_embs, val_conds], dim=1)
    
    model_preds = {
        "ESM-2 embedding + MLP": [],
        "Condition-aware MLP": [],
        "DiG-inspired energy-based baseline": [],
        "active-learning baseline (inspired by condensate active-ML)": [],
        "Biophysical Heuristic Floor (catGRANULE/FuzDrop proxy)": []
    }
    
    with open(os.path.join(BASE_DIR, "PRE_REGISTRATION.json"), "r", encoding="utf-8") as f:
        config_hash = json.load(f)["config_hash"]
    
    print("\nEvaluating Biophysical Heuristic Floor...")
    heuristic_preds = np.array([compute_biophysical_score(seq) for seq in df_val['Sequence']])
    for seed in SEEDS:
        model_preds["Biophysical Heuristic Floor (catGRANULE/FuzDrop proxy)"].append(heuristic_preds)
        
    for seed in SEEDS:
        print(f"\nTraining Models for Seed {seed}...")
        
        # 1. ESM-2 + MLP
        run_id = f"esm2_mlp_seed{seed}"
        preds = train_model("vanilla", X_train_seq, y_train, X_val_seq, y_val, seed, run_id, config_hash)
        model_preds["ESM-2 embedding + MLP"].append(preds)
        
        # 2. Condition-aware MLP
        run_id = f"condition_mlp_seed{seed}"
        preds = train_model("vanilla", X_train_cond, y_train, X_val_cond, y_val, seed, run_id, config_hash)
        model_preds["Condition-aware MLP"].append(preds)
        
        # 3. DiG-inspired EBM
        run_id = f"dig_ebm_seed{seed}"
        preds = train_model("energy_based", X_train_cond, y_train, X_val_cond, y_val, seed, run_id, config_hash)
        model_preds["DiG-inspired energy-based baseline"].append(preds)
        
        # 4. Active-learning baseline
        run_id = f"active_learning_seed{seed}"
        preds = train_active_learning(X_train_cond, y_train, X_val_cond, y_val, seed, run_id, config_hash)
        model_preds["active-learning baseline (inspired by condensate active-ML)"].append(preds)

    print("\nTraining complete. Calculating bootstrap CIs on validation set...")
    
    results = {}
    for model_name, preds_list in model_preds.items():
        avg_preds = np.mean(preds_list, axis=0)
        df_eval = pd.DataFrame({
            "cluster_id": df_val["cluster_id"],
            "label": df_val["label"],
            "pred": avg_preds
        })
        ci_results = cluster_block_bootstrap_ci(df_eval, n_iterations=1000)
        results[model_name] = ci_results
        print(f"\n{model_name}:")
        print(f"  AUPRC: {ci_results['AUPRC_median']:.4f} (95% CI: [{ci_results['AUPRC_ci'][0]:.4f}, {ci_results['AUPRC_ci'][1]:.4f}])")
        print(f"  AUROC: {ci_results['AUROC_median']:.4f} (95% CI: [{ci_results['AUROC_ci'][0]:.4f}, {ci_results['AUROC_ci'][1]:.4f}])")

    print("\nPerforming empirical-variance power re-check...")
    best_baseline_preds = np.mean(model_preds["Condition-aware MLP"], axis=0)
    df_best = pd.DataFrame({
        "cluster_id": df_val["cluster_id"],
        "label": df_val["label"],
        "pred": best_baseline_preds
    })
    
    family_auprcs = []
    for fid, group in df_best.groupby('cluster_id'):
        if len(group) >= 5 and len(group['label'].unique()) == 2:
            family_auprcs.append(compute_auprc(group['label'].values, group['pred'].values))
            
    empirical_sigma = np.std(family_auprcs) if len(family_auprcs) > 1 else 0.10
    print(f"Empirical per-family AUPRC standard deviation (spread): {empirical_sigma:.4f}")
    
    test_families_n = 49
    test_records_n = 269
    emp_power, emp_mde = simulate_empirical_power(
        test_families_n, test_records_n, empirical_sigma, true_delta=0.05
    )
    
    print(f"Empirical Power at delta=0.05 (test set): {emp_power*100:.1f}%")
    print(f"Empirical Minimum Detectable Effect (80% power): {emp_mde:.3f} AUPRC")
    
    results_output = {
        "run_metadata": {
            "run_id": "run_gate1_freeze_20260612",
            "config_hash": config_hash,
            "esm_backbone": "facebook/esm2_t33_650M_UR50D"
        },
        "baselines": results,
        "empirical_power_check": {
            "empirical_sigma_family": float(empirical_sigma),
            "test_set_families": test_families_n,
            "test_set_records": test_records_n,
            "power_at_0_05": float(emp_power),
            "minimum_detectable_effect_80_power": float(emp_mde),
            "statement": "Power is sufficient if MDE is <= 0.08, otherwise flagged for discussion."
        },
        "not_reproduced_baselines": {
            "DiG-style equilibrium-distribution baseline": (
                "DiG (Distributional Graphormer, Nat. Mach. Intell. 2024) is a 3D molecular structure ensemble "
                "predictor, not a sequence-based binary LLPS classifier. No public code/description exists for "
                "sequence-based binary LLPS classification. Relabeled and implemented a 'DiG-inspired energy-based baseline' instead."
            )
        }
    }
    
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results_output, f, indent=2, ensure_ascii=False)
        
    print(f"\nAll results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
