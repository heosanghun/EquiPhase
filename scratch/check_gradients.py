import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import sys

# Ensure workspace is in path
sys.path.append("d:/AI/EquiPhase")

from iss_data import FoldSwitchDataset, split_dataset_by_family, collate_fn
from equiphase.models.symplectic_deq import SymplecticDEQ
from equiphase.models.losses import MasterpieceLoss

def check_grads():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Initialize a small batch
    df = pd.read_csv("data/mutations.csv")
    sequences = df["sequence"].tolist()[:4]
    delta_ddgs = df["delta_ddg"].tolist()[:4]
    fold_family_ids = df["fold_family_id"].tolist()[:4]
    pdb_ids = df["pdb_id"].tolist()[:4]
    
    dataset = FoldSwitchDataset(
        sequences=sequences,
        control_params=delta_ddgs,
        delta_ddgs=delta_ddgs,
        fold_family_ids=fold_family_ids,
        pdb_ids=pdb_ids,
        esm_dim=1280
    )
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))
    padded_X, lams, padded_targets_A, padded_targets_B, ddgs, families, mut_indices, padded_X_wt = batch
    
    padded_X = padded_X.to(device)
    lams = lams.to(device)
    ddgs = ddgs.to(device)
    mut_indices = mut_indices.to(device)
    padded_X_wt = padded_X_wt.to(device)
    
    # Initialize model
    model = SymplecticDEQ(
        esm_dim=1280,
        latent_dim=64,
        num_starts=2,
        dt=0.05,
        damping=0.2
    ).to(device)
    
    criterion = MasterpieceLoss(tau=0.1, gamma=2.0, w_repulsive=2.0, w_anchor=0.5, w_switch=1.0).to(device)
    
    # 1. Forward pass
    z_star, margins, coords_pred = model(padded_X, lams, mut_indices=mut_indices, X_wt_esm=padded_X_wt)
    t_A = padded_targets_A.to(device)
    t_B = padded_targets_B.to(device)
    
    # We want to check gradients from loss_switch only
    loss_soft_min, loss_rep, loss_anchor, loss_switch, total_loss = 0.0, 0.0, 0.0, 0.0, 0.0
    
    total_loss, loss_dict = criterion(coords_pred, t_A, t_B, z_star, margins=margins, delta_delta_g=ddgs)
    
    print("\nLoss values:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
        
    # Check gradients from loss_switch only
    model.zero_grad()
    loss_dict_switch = loss_dict["loss_switch"]
    loss_switch_tensor = criterion.w_switch * loss_dict["loss_switch"]
    
    # Since loss_dict is a dict of floats, we must compute it directly to get the tensor
    delta_m = margins[:, 1] - margins[:, 0]
    loss_switch_val = torch.mean((4.0 * delta_m + ddgs.squeeze(-1))**2)
    
    print("\nComputing gradients for loss_switch...")
    loss_switch_val.backward(retain_graph=True)
    
    print("\nGradients of force_net parameters:")
    for name, param in model.named_parameters():
        if "force_net" in name:
            if param.grad is not None:
                grad_mean = param.grad.abs().mean().item()
                grad_max = param.grad.abs().max().item()
                print(f"  {name}: mean = {grad_mean:.4e}, max = {grad_max:.4e}")
            else:
                print(f"  {name}: None")
                
    # Check gradients from loss_soft_min only
    model.zero_grad()
    # Let's compute soft_min loss directly to get tensor
    mse_A_list = []
    mse_B_list = []
    from equiphase.models.losses import stable_cdist
    D_A = stable_cdist(t_A, t_A, eps=criterion.eps)
    D_B = stable_cdist(t_B, t_B, eps=criterion.eps)
    for k in range(model.num_starts):
        coords_k = coords_pred[:, k, :, :]
        D_k = stable_cdist(coords_k, coords_k, eps=criterion.eps)
        mse_A_list.append(torch.mean((D_k - D_A)**2, dim=(1, 2)))
        mse_B_list.append(torch.mean((D_k - D_B)**2, dim=(1, 2)))
    mse_A_stack = torch.stack(mse_A_list, dim=1)
    mse_B_stack = torch.stack(mse_B_list, dim=1)
    L_soft_min_A = - criterion.tau * torch.logsumexp(-mse_A_stack / criterion.tau, dim=1)
    L_soft_min_B = - criterion.tau * torch.logsumexp(-mse_B_stack / criterion.tau, dim=1)
    loss_soft_min_val = torch.mean(L_soft_min_A + L_soft_min_B)
    
    print("\nComputing gradients for loss_soft_min...")
    loss_soft_min_val.backward()
    
    print("\nGradients of force_net parameters (from soft_min):")
    for name, param in model.named_parameters():
        if "force_net" in name:
            if param.grad is not None:
                grad_mean = param.grad.abs().mean().item()
                grad_max = param.grad.abs().max().item()
                print(f"  {name}: mean = {grad_mean:.4e}, max = {grad_max:.4e}")
            else:
                print(f"  {name}: None")

if __name__ == "__main__":
    check_grads()
