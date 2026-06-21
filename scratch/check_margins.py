import torch
import pandas as pd
import numpy as np
import sys

# Ensure workspace is in path
sys.path.append("d:/AI/EquiPhase")

from iss_data import FoldSwitchDataset, collate_fn
from equiphase.models.symplectic_deq import SymplecticDEQ

def check_margins():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df = pd.read_csv("data/mutations.csv")
    sequences = df["sequence"].tolist()[:10]
    delta_ddgs = df["delta_ddg"].tolist()[:10]
    fold_family_ids = df["fold_family_id"].tolist()[:10]
    pdb_ids = df["pdb_id"].tolist()[:10]
    
    dataset = FoldSwitchDataset(
        sequences=sequences,
        control_params=delta_ddgs,
        delta_ddgs=delta_ddgs,
        fold_family_ids=fold_family_ids,
        pdb_ids=pdb_ids,
        esm_dim=1280
    )
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=10, collate_fn=collate_fn)
    batch = next(iter(loader))
    padded_X, lams, padded_targets_A, padded_targets_B, ddgs, families, mut_indices, padded_X_wt = batch
    
    padded_X = padded_X.to(device)
    lams = lams.to(device)
    ddgs = ddgs.to(device)
    mut_indices = mut_indices.to(device)
    padded_X_wt = padded_X_wt.to(device)
    
    model = SymplecticDEQ(
        esm_dim=1280,
        latent_dim=64,
        num_starts=2,
        dt=0.05,
        damping=0.2
    ).to(device)
    
    with torch.no_grad():
        z_star, margins, coords_pred = model(padded_X, lams, mut_indices=mut_indices, X_wt_esm=padded_X_wt)
        
    print("Sample | ddG (Lam) | Margin 0 (Fold A) | Margin 1 (Fold B) | Diff (M1 - M0)")
    print("-" * 75)
    for i in range(10):
        print(f"  {i:02d}   |  {ddgs[i].item():.4f}  |      {margins[i, 0].item():.4f}       |      {margins[i, 1].item():.4f}       |   {margins[i, 1].item() - margins[i, 0].item():.4f}")

if __name__ == "__main__":
    check_margins()
