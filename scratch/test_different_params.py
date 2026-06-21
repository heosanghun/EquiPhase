import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import sys

# Ensure workspace is in path
sys.path.append("d:/AI/EquiPhase")
sys.path.append("/workspace")

from iss_data import FoldSwitchDataset, split_dataset_by_family, collate_fn
from equiphase.models.symplectic_deq import SymplecticDEQ
from equiphase.models.losses import MasterpieceLoss
from iss_train import ISSTrainer

def run_test_param(w_switch=1.0, force_scale=0.1, lr=1e-3, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTesting: w_switch={w_switch}, force_scale={force_scale}, lr={lr}, epochs={epochs}")
    
    # Load dataset
    df = pd.read_csv("data/mutations.csv")
    sequences = df["sequence"].tolist()
    delta_ddgs = df["delta_ddg"].tolist()
    fold_family_ids = df["fold_family_id"].tolist()
    pdb_ids = df["pdb_id"].tolist()
    
    # Parse coordinates
    target_structures_A = []
    target_structures_B = []
    for pdb_id in pdb_ids:
        pdb_A_path = f"data/pdbs/{pdb_id}_A.pdb"
        pdb_B_path = f"data/pdbs/{pdb_id}_B.pdb"
        _, coords_A = parse_pdb(pdb_A_path)
        _, coords_B = parse_pdb(pdb_B_path)
        target_structures_A.append(torch.tensor(coords_A, dtype=torch.float32))
        target_structures_B.append(torch.tensor(coords_B, dtype=torch.float32))
        
    dataset = FoldSwitchDataset(
        sequences=sequences,
        control_params=delta_ddgs,
        delta_ddgs=delta_ddgs,
        fold_family_ids=fold_family_ids,
        pdb_ids=pdb_ids,
        esm_dim=1280
    )
    dataset.target_structures_A = target_structures_A
    dataset.target_structures_B = target_structures_B
    
    train_subset, val_subset, test_subset = split_dataset_by_family(
        dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42
    )
    
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    model = SymplecticDEQ(
        esm_dim=1280,
        latent_dim=64,
        num_starts=2,
        dt=0.05,
        damping=0.2
    ).to(device)
    
    # Patch force_forward to use the force_scale parameter
    original_force_forward = model.force_forward
    def patched_force_forward(q, X_pooled, lam_eff, X_mut=None, X_wt_res=None):
        lam_emb = model.lam_proj(lam_eff)
        inputs = torch.cat([q, X_pooled, lam_emb], dim=-1)
        force = force_scale * torch.tanh(model.force_net(inputs))
        
        if X_mut is None:
            X_mut = X_pooled
        if X_wt_res is None:
            X_wt_res = X_pooled
            
        seq_mod = 0.1 * torch.tanh(model.seq_proj_q(X_mut - X_wt_res)) * torch.tanh(q)
        force = force + seq_mod
        
        bilinear_term = lam_eff * torch.tanh(model.bilinear_proj_q(q))
        force = force + bilinear_term
        
        return force
    model.force_forward = patched_force_forward
    
    criterion = MasterpieceLoss(tau=0.1, gamma=2.0, w_repulsive=2.0, w_anchor=0.5, w_switch=w_switch).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    trainer = ISSTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )
    
    # Train
    for epoch in range(1, epochs + 1):
        loss, loss_dict = trainer.train_epoch()
        print(f"  Epoch {epoch:02} | Loss: {loss:.4f} | Switch Loss: {loss_dict['loss_switch']:.4f} | SoftMin: {loss_dict['loss_soft_min']:.4f}")
        
    # Evaluate on val
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in val_loader:
            padded_X, lams, padded_targets_A, padded_targets_B, ddgs, _, mut_indices, padded_X_wt = batch
            padded_X = padded_X.to(device)
            lams = lams.to(device)
            mut_indices = mut_indices.to(device)
            padded_X_wt = padded_X_wt.to(device)
            
            _, margins, _ = model(padded_X, lams, mut_indices=mut_indices, X_wt_esm=padded_X_wt)
            # We predict using margins[:, 0] - margins[:, 1]
            pred_score = (margins[:, 0] - margins[:, 1]).cpu().numpy()
            all_preds.extend(pred_score)
            binary_target = (lams.squeeze(-1) > 0.5).long().cpu().numpy()
            all_targets.extend(binary_target)
            
    from equiphase.eval.audit_protocol import compute_auroc
    auroc = compute_auroc(all_targets, all_preds)
    print(f"  Validation Naive AUROC: {auroc:.4f}")
    
    # Print sample margins
    print("  Sample margins:")
    with torch.no_grad():
        batch = next(iter(val_loader))
        padded_X, lams, _, _, ddgs, _, mut_indices, padded_X_wt = batch
        padded_X = padded_X.to(device)
        lams = lams.to(device)
        mut_indices = mut_indices.to(device)
        padded_X_wt = padded_X_wt.to(device)
        _, margins, _ = model(padded_X, lams, mut_indices=mut_indices, X_wt_esm=padded_X_wt)
        for i in range(min(5, len(ddgs))):
            print(f"    ddG={ddgs[i].item():.2f} | margins=[{margins[i, 0].item():.4f}, {margins[i, 1].item():.4f}]")

def parse_pdb(pdb_path):
    from iss_data import parse_pdb as original_parse
    return original_parse(pdb_path)

if __name__ == "__main__":
    run_test_param(w_switch=100.0, force_scale=1.0, lr=1e-3, epochs=5)
    run_test_param(w_switch=500.0, force_scale=2.0, lr=5e-3, epochs=5)
