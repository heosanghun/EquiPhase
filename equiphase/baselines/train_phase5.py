import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
import time

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

BASE_DIR = "D:/AI/EquiPhase/"
DATA_DIR = os.path.join(BASE_DIR, 'equiphase', 'data')
RESULTS_PATH = os.path.join(BASE_DIR, "baselines_results_phase5.json")

sys.path.append(BASE_DIR)
from equiphase.eval.metrics import compute_metrics, cluster_block_bootstrap_ci, check_and_register_run

EPOCHS = 25
BATCH_SIZE = 64
LR = 1e-3
SEEDS = [42, 100, 2026, 777, 999]

class LLPSResidueDataset(Dataset):
    def __init__(self, df, residue_embs, biophys_embs, normalize_fn):
        self.df = df
        self.residue_embs = residue_embs
        self.biophys_embs = biophys_embs
        self.conds = normalize_fn(df)
        self.labels = df['label'].values
        self.sequences = df['Sequence'].values
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        emb = self.residue_embs[seq] # (L, 1280)
        biophys = self.biophys_embs[seq] # (10,)
        cond = self.conds[idx] # (4,)
        label = self.labels[idx]
        return emb, biophys, cond, label

def collate_fn(batch):
    embs, biophyss, conds, labels = zip(*batch)
    lengths = [len(e) for e in embs]
    max_len = max(lengths)
    
    padded_embs = []
    masks = []
    for e in embs:
        L_i = len(e)
        padded = np.zeros((max_len, 1280), dtype=np.float32)
        padded[:L_i] = e
        padded_embs.append(padded)
        
        mask = np.zeros(max_len, dtype=bool)
        mask[:L_i] = True
        masks.append(mask)
        
    padded_embs = torch.tensor(np.stack(padded_embs), dtype=torch.float32)
    masks = torch.tensor(np.stack(masks), dtype=torch.bool)
    biophyss = torch.tensor(np.stack(biophyss), dtype=torch.float32)
    conds = torch.tensor(np.stack(conds), dtype=torch.float32)
    labels = torch.tensor(np.array(labels), dtype=torch.float32)
    
    return padded_embs, masks, biophyss, conds, labels

class AttentionPooling(nn.Module):
    def __init__(self, d_model=1280):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model))
        self.key_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask):
        # x: (B, L, D)
        # mask: (B, L)
        keys = self.key_proj(x) # (B, L, D)
        scores = torch.matmul(keys, self.query) # (B, L)
        scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=-1) # (B, L)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1) # (B, D)
        return pooled, weights

class AttentionMLP(nn.Module):
    def __init__(self, emb_dim=1280, biophys_dim=10, cond_dim=4):
        super().__init__()
        self.pool = AttentionPooling(emb_dim)
        self.fc1 = nn.Linear(emb_dim + biophys_dim + cond_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, mask, biophys, cond):
        pooled, weights = self.pool(x, mask)
        x_cat = torch.cat([pooled, biophys, cond], dim=1)
        x_fc = F.relu(self.fc1(x_cat))
        x_fc = self.dropout(x_fc)
        x_fc = F.relu(self.fc2(x_fc))
        x_fc = self.dropout(x_fc)
        return torch.sigmoid(self.fc3(x_fc)).squeeze(-1), pooled, weights

def normalize_conditions(df):
    log_c = np.log10(df['parsed_c_sat'].values + 1e-3)
    log_c_norm = (log_c - np.mean(log_c)) / (np.std(log_c) + 1e-5)
    salt_norm = df['parsed_salt'].values / 1000.0
    ph_norm = df['parsed_ph'].values / 14.0
    temp_norm = df['parsed_temp'].values / 100.0
    conds = np.stack([log_c_norm, salt_norm, ph_norm, temp_norm], axis=1)
    return torch.tensor(conds, dtype=torch.float32)

def train_mlp_seed(seed, train_loader, val_loader, device, run_id, config_hash):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = AttentionMLP().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_val_auprc = 0.0
    best_state = None
    
    loss_trace = []
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        for batch_emb, batch_mask, batch_bio, batch_cond, batch_y in train_loader:
            batch_emb, batch_mask = batch_emb.to(device), batch_mask.to(device)
            batch_bio, batch_cond, batch_y = batch_bio.to(device), batch_cond.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            pred, _, _ = model(batch_emb, batch_mask, batch_bio, batch_cond)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            
        loss_trace.append(np.mean(epoch_losses))
        
        # Eval
        model.eval()
        val_preds = []
        val_ys = []
        with torch.no_grad():
            for batch_emb, batch_mask, batch_bio, batch_cond, batch_y in val_loader:
                batch_emb, batch_mask = batch_emb.to(device), batch_mask.to(device)
                batch_bio, batch_cond = batch_bio.to(device), batch_cond.to(device)
                pred, _, _ = model(batch_emb, batch_mask, batch_bio, batch_cond)
                val_preds.extend(pred.cpu().numpy())
                val_ys.extend(batch_y.numpy())
                
        metrics = compute_metrics(np.array(val_ys), np.array(val_preds))
        if metrics["AUPRC"] > best_val_auprc:
            best_val_auprc = metrics["AUPRC"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
    # Load best model
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    # R3 integrity check
    check_and_register_run(run_id, config_hash, {"AUPRC": best_val_auprc}, training_loss_trace=loss_trace)
    return model

def main():
    print("Loading datasets and cached features...")
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train_phase5.tsv"), sep="\t")
    df_val = pd.read_csv(os.path.join(DATA_DIR, "val_phase5.tsv"), sep="\t")
    
    with open(os.path.join(DATA_DIR, "esm2_residue_embeddings.pkl"), "rb") as f:
        residue_embs = pickle.load(f)
    with open(os.path.join(DATA_DIR, "biophysical_features.pkl"), "rb") as f:
        biophys_features = pickle.load(f)
        
    with open(os.path.join(BASE_DIR, "PRE_REGISTRATION_PHASE5.json"), "r", encoding="utf-8") as f:
        pre_reg = json.load(f)
        config_hash = pre_reg["config_hash"]
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Dataset and DataLoader
    train_dataset = LLPSResidueDataset(df_train, residue_embs, biophys_features, normalize_conditions)
    val_dataset = LLPSResidueDataset(df_val, residue_embs, biophys_features, normalize_conditions)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    mlp_val_preds = []
    xgb_val_preds = []
    
    # For extracting pooled embeddings
    all_train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    for seed in SEEDS:
        print(f"\n--- Training Seed {seed} ---")
        # 1. Train AttentionMLP
        run_id = f"phase5_mlp_seed{seed}"
        model = train_mlp_seed(seed, train_loader, val_loader, device, run_id, config_hash)
        
        # Predict on validation
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch_emb, batch_mask, batch_bio, batch_cond, _ in val_loader:
                batch_emb, batch_mask = batch_emb.to(device), batch_mask.to(device)
                batch_bio, batch_cond = batch_bio.to(device), batch_cond.to(device)
                pred, _, _ = model(batch_emb, batch_mask, batch_bio, batch_cond)
                val_preds.extend(pred.cpu().numpy())
        mlp_val_preds.append(np.array(val_preds))
        
        # 2. Extract attention-pooled embeddings to train XGBoost
        train_pooled = []
        train_labels = []
        with torch.no_grad():
            for batch_emb, batch_mask, batch_bio, batch_cond, batch_y in all_train_loader:
                batch_emb, batch_mask = batch_emb.to(device), batch_mask.to(device)
                batch_bio, batch_cond = batch_bio.to(device), batch_cond.to(device)
                _, pooled, _ = model(batch_emb, batch_mask, batch_bio, batch_cond)
                # Concatenate pooled (1280) + biophys (10) + cond (4)
                X_batch = torch.cat([pooled, batch_bio, batch_cond], dim=1)
                train_pooled.append(X_batch.cpu().numpy())
                train_labels.extend(batch_y.numpy())
        X_train_xgb = np.concatenate(train_pooled, axis=0)
        y_train_xgb = np.array(train_labels)
        
        val_pooled = []
        with torch.no_grad():
            for batch_emb, batch_mask, batch_bio, batch_cond, _ in val_loader:
                batch_emb, batch_mask = batch_emb.to(device), batch_mask.to(device)
                batch_bio, batch_cond = batch_bio.to(device), batch_cond.to(device)
                _, pooled, _ = model(batch_emb, batch_mask, batch_bio, batch_cond)
                X_batch = torch.cat([pooled, batch_bio, batch_cond], dim=1)
                val_pooled.append(X_batch.cpu().numpy())
        X_val_xgb = np.concatenate(val_pooled, axis=0)
        
        # 3. Train Tab-Monotone XGBoost
        # Features layout:
        # Index 0 to 1279: pooled embedding (1280)
        # Index 1280 to 1289: biophysical descriptors (10)
        # Index 1290: normalized solute concentration (log_c_norm)
        # Index 1291: normalized salt concentration
        # Index 1292: normalized pH
        # Index 1293: normalized temperature
        
        # Enforce positive monotonic constraint on solute concentration (index 1290)
        constraints = [0] * 1294
        constraints[1290] = 1 # Higher concentration => higher or equal probability of LLPS
        monotone_constraints = tuple(constraints)
        
        xgb_model = xgb.XGBClassifier(
            max_depth=5,
            learning_rate=0.05,
            n_estimators=150,
            monotone_constraints=monotone_constraints,
            random_state=seed,
            eval_metric="logloss"
        )
        xgb_model.fit(X_train_xgb, y_train_xgb)
        
        # Predict on validation
        preds_xgb = xgb_model.predict_proba(X_val_xgb)[:, 1]
        xgb_val_preds.append(preds_xgb)
        
    print("\nTraining complete. Running bootstrap evaluation on validation set...")
    
    # Average predictions over seeds
    avg_mlp_preds = np.mean(mlp_val_preds, axis=0)
    avg_xgb_preds = np.mean(xgb_val_preds, axis=0)
    
    df_mlp = pd.DataFrame({
        "cluster_id": df_val["cluster_id"],
        "label": df_val["label"],
        "pred": avg_mlp_preds
    })
    
    df_xgb = pd.DataFrame({
        "cluster_id": df_val["cluster_id"],
        "label": df_val["label"],
        "pred": avg_xgb_preds
    })
    
    mlp_ci = cluster_block_bootstrap_ci(df_mlp, n_iterations=1000)
    xgb_ci = cluster_block_bootstrap_ci(df_xgb, n_iterations=1000)
    
    print("\n================== Results Summary ==================")
    print("AttentionMLP (ESM-2 + Biophysical + Attention):")
    print(f"  AUPRC: {mlp_ci['AUPRC_median']:.4f} (95% CI: [{mlp_ci['AUPRC_ci'][0]:.4f}, {mlp_ci['AUPRC_ci'][1]:.4f}])")
    print(f"  AUROC: {mlp_ci['AUROC_median']:.4f} (95% CI: [{mlp_ci['AUROC_ci'][0]:.4f}, {mlp_ci['AUROC_ci'][1]:.4f}])")
    
    print("\nTab-Monotone XGBoost:")
    print(f"  AUPRC: {xgb_ci['AUPRC_median']:.4f} (95% CI: [{xgb_ci['AUPRC_ci'][0]:.4f}, {xgb_ci['AUPRC_ci'][1]:.4f}])")
    print(f"  AUROC: {xgb_ci['AUROC_median']:.4f} (95% CI: [{xgb_ci['AUROC_ci'][0]:.4f}, {xgb_ci['AUROC_ci'][1]:.4f}])")
    
    # Check if they clear the validation no-skill baseline (0.6811)
    base_rate = float(df_val['label'].mean())
    print(f"\nValidation No-Skill Baseline: {base_rate:.4f}")
    
    mlp_clears = mlp_ci['AUPRC_ci'][0] > base_rate
    xgb_clears = xgb_ci['AUPRC_ci'][0] > base_rate
    
    print(f"AttentionMLP clears baseline?: {mlp_clears}")
    print(f"Tab-Monotone XGBoost clears baseline?: {xgb_clears}")
    
    # Save results to json
    results_output = {
        "run_metadata": {
            "run_id": "run_phase5_h2_20260612",
            "config_hash": config_hash
        },
        "validation_no_skill_baseline": base_rate,
        "models": {
            "AttentionMLP": mlp_ci,
            "Tab-Monotone XGBoost": xgb_ci
        },
        "win_status": {
            "AttentionMLP_win": mlp_clears,
            "Tab-Monotone XGBoost_win": xgb_clears
        }
    }
    
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results_output, f, indent=2)
    print(f"Results successfully written to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
