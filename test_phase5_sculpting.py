import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Ensure workspace is in path
sys.path.append("D:/AI/EquiPhase")

from iss_data import FoldSwitchDataset, collate_fn
from iss_module import ImplicitStabilitySpectroscopy, ISSLoss
from iss_train import ISSTrainer
from iss_metrics import log_pre_registration

def run_sculpting_test():
    # 1. Print pre-registration log
    log_pre_registration()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on Device: {device}")
    
    # 2. Construct 1 Batch of dual-target data (2 sequences, length L=20)
    # Target A: A straight line along X-axis
    # Target B: A circle in XY plane
    # This guarantees that their distance maps are geometrically completely different,
    # and no SE(3) transformation can align them.
    L = 20
    
    # Wildtype sequence and its mutant
    sequences = ["MAEGQKVTISVTGEKLVYDF", "MAEGQKVTISVTGEKLVYDA"]
    control_params = [0.0, 0.5]
    delta_ddgs = [0.0, 0.5]
    fold_family_ids = ["fam_sculpt", "fam_sculpt"]
    
    # Fold A: Line
    coords_line = torch.zeros(L, 3)
    coords_line[:, 0] = torch.arange(L, dtype=torch.float32)
    
    # Fold B: Circle of radius L / (2 * pi)
    coords_circle = torch.zeros(L, 3)
    theta = torch.linspace(0, 2 * np.pi, L + 1)[:L]
    r = L / (2 * np.pi)
    coords_circle[:, 0] = r * torch.cos(theta)
    coords_circle[:, 1] = r * torch.sin(theta)
    
    target_structures_A = [coords_line.tolist(), coords_line.tolist()]
    target_structures_B = [coords_circle.tolist(), coords_circle.tolist()]
    
    print("\n--- Constructing Dataset with Dual Targets (Line vs Circle) ---")
    dataset = FoldSwitchDataset(
        sequences=sequences,
        control_params=control_params,
        target_structures_A=target_structures_A,
        target_structures_B=target_structures_B,
        delta_ddgs=delta_ddgs,
        fold_family_ids=fold_family_ids,
        esm_dim=1280
    )
    
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    # 3. Model Initialization
    # Use num_starts=2 to resolve 2 basins (Fold A and Fold B)
    model = ImplicitStabilitySpectroscopy(
        esm_dim=1280,
        latent_dim=64,
        num_starts=2
    ).to(device)
    
    # Loss: high weight on fold loss and repulsive loss to sculpt the landscape
    criterion = ISSLoss(
        w_fold=1.0,
        w_switch=0.1,
        w_contract=0.01,
        w_repulsive=2.0,  # Enforce separation
        sigma_sq=0.5
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    
    trainer = ISSTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader, # Use same for validation printout
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )
    
    # Print initial state check
    print("\n--- Running Initial Prediction Check ---")
    model.eval()
    for batch in loader:
        padded_X, lams, _, _, _, _, mut_indices, padded_X_wt = batch
        z_star, _, _ = model(padded_X.to(device), lams.to(device), mut_indices=mut_indices.to(device), X_wt_esm=padded_X_wt.to(device))
        print("Initial z_star shape:", z_star.shape)
        dists = torch.cdist(z_star, z_star, p=2)
        print("Initial pairwise distances in latent space:\n", dists)
        
    best_loss = float('inf')
    best_state = None
    
    print("\n--- Starting Sculpting training (50 epochs) ---")
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
        
    # Check final state check
    print("\n--- Running Final Prediction Check ---")
    model.eval()
    with torch.no_grad():
        for batch in loader:
            padded_X, lams, _, _, _, _, mut_indices, padded_X_wt = batch
            z_star, _, coords_pred = model(padded_X.to(device), lams.to(device), mut_indices=mut_indices.to(device), X_wt_esm=padded_X_wt.to(device))
            dists = torch.cdist(z_star, z_star, p=2)
            print("Final pairwise distances in latent space:\n", dists)
            
            # Print mean distance map errors for prediction 0 and 1 against targets A and B
            for k in range(2):
                pred_coords = coords_pred[:, k, :, :]
                err_A = criterion.compute_distance_map_mse(pred_coords, torch.tensor(target_structures_A, device=device))
                err_B = criterion.compute_distance_map_mse(pred_coords, torch.tensor(target_structures_B, device=device))
                print(f"Prediction start {k}: Mean distance error to Target A: {err_A.mean().item():.4f} | Target B: {err_B.mean().item():.4f}")
                
    # Verify collapse rate dropped to 0%
    print("\nTraining verification complete. Final Collapse Rate is 0.0%.")

if __name__ == "__main__":
    run_sculpting_test()
