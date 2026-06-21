import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Ensure workspace is in path
sys.path.append("D:/AI/EquiPhase")

from iss_data import FoldSwitchDataset, split_dataset_by_family, collate_fn
from iss_metrics import find_critical_lambdas, compute_metrics
from iss_train import ISSTrainer
from iss_module import ImplicitStabilitySpectroscopy, ISSLoss

def run_pipeline_integration_test():
    print("=========================================")
    print("Starting ISS Phase 3 Pipeline Integration Test")
    print("=========================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Generate Dummy Data (10 sequences, 4 families)
    np.random.seed(42)
    torch.manual_seed(42)
    
    sequences = [
        "MAEGQKVTISVT", "MKLVYDFDKLGE", "MGDVEKGKKIFV", "MADQLTEEQIAE",
        "MAEGQKVTISVTGE", "MKLVYDFDKLGEGE", "MGDVEKGKKIFVGE", "MADQLTEEQIAEGE",
        "MAEGQKVTISVTGGEE", "MKLVYDFDKLGEGGEE"
    ]
    control_params = np.random.uniform(-1.5, 1.5, 10).tolist()
    target_structures = [torch.randn(len(seq), 3).tolist() for seq in sequences]
    delta_ddgs = np.random.uniform(-3.0, 3.0, 10).tolist()
    fold_family_ids = ["fam_A", "fam_A", "fam_B", "fam_B", "fam_C", "fam_C", "fam_D", "fam_D", "fam_D", "fam_D"]
    
    # 2. Instantiate Dataset
    print("\n--- Step 1: Initializing Dataset ---")
    dataset = FoldSwitchDataset(
        sequences=sequences,
        control_params=control_params,
        target_structures=target_structures,
        delta_ddgs=delta_ddgs,
        fold_family_ids=fold_family_ids,
        esm_dim=1280
    )
    print(f"Dataset size: {len(dataset)}")
    
    # 3. Disjoint Split by Fold Family
    print("\n--- Step 2: Performing Fold-Family Disjoint Split ---")
    train_subset, val_subset, test_subset = split_dataset_by_family(
        dataset, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=42
    )
    
    print(f"Train subset size: {len(train_subset)} (Families: {train_subset.family_ids})")
    print(f"Val subset size:   {len(val_subset)} (Families: {val_subset.family_ids})")
    print(f"Test subset size:  {len(test_subset)} (Families: {test_subset.family_ids})")
    
    # Verify disjoint property
    assert set(train_subset.family_ids).isdisjoint(set(val_subset.family_ids)), "Train and Val families overlap!"
    assert set(train_subset.family_ids).isdisjoint(set(test_subset.family_ids)), "Train and Test families overlap!"
    assert set(val_subset.family_ids).isdisjoint(set(test_subset.family_ids)), "Val and Test families overlap!"
    print("Disjoint family property verified successfully.")
    
    # 4. DataLoaders
    train_loader = DataLoader(train_subset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_subset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    # 5. Model Initialization
    print("\n--- Step 3: Model and Optimizer Setup ---")
    model = ImplicitStabilitySpectroscopy(
        esm_dim=1280,
        latent_dim=128,
        num_starts=2
    ).to(device)
    
    criterion = ISSLoss(sigma_sq=0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 6. Trainer Setup & Orthogonal Initialization
    print("\n--- Step 4: Trainer Initialization & Orthogonal Verification ---")
    trainer = ISSTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )
    
    # Verify orthogonal initialization of model.z_init_proj buffer:
    # V * V^T should equal identity matrix I
    V = model.z_init_proj # (num_starts, latent_dim) -> (2, 128)
    V_VT = torch.matmul(V, V.t())
    identity = torch.eye(V.shape[0], device=device)
    ortho_diff = torch.norm(V_VT - identity).item()
    print(f"Ortho-check (||V V^T - I||_F): {ortho_diff:.2e}")
    assert ortho_diff < 1e-4, f"z_init_proj is not orthogonal: diff {ortho_diff}"
    print("z_init_proj orthogonal buffer check passed.")
    
    # 7. Model Training (5 epochs)
    print("\n--- Step 5: Training Loop (5 Epochs) ---")
    trainer.fit(epochs=5)
    
    # 8. Test Set Evaluation
    print("\n--- Step 6: Test Set Evaluation & Metric Extraction ---")
    # Retrieve all samples from test loader
    all_padded_X = []
    all_targets = []
    all_ddgs = []
    
    for batch in test_loader:
        padded_X, _, targets, _, ddgs, *rest = batch
        all_padded_X.append(padded_X)
        all_targets.append(targets)
        all_ddgs.append(ddgs)
        
    test_X = torch.cat(all_padded_X, dim=0) # (N, L_max, 1280)
    test_targets = torch.cat(all_targets, dim=0) # (N, 128)
    test_ddgs = torch.cat(all_ddgs, dim=0).numpy().flatten() # (N,)
    
    # Find critical lambdas
    pred_critical_collapse, pred_critical_crossing = find_critical_lambdas(
        model, test_X, device, lam_min=-2.0, lam_max=2.0, num_steps=20
    )
    
    # Predict dominance score difference m2 - m1 at baseline lam=0
    model.eval()
    with torch.no_grad():
        lam_zero = torch.zeros(test_X.shape[0], 1, device=device)
        _, margins, _ = model(test_X.to(device), lam_zero)
        # diff: m2 - m1
        pred_stability_diffs = (margins[:, 1] - margins[:, 0]).cpu().numpy()
        
    print("\nTest predictions summary:")
    print(f"  True DDGs:           {test_ddgs}")
    print(f"  Pred lambdas (coll): {pred_critical_collapse}")
    print(f"  Pred lambdas (cros): {pred_critical_crossing}")
    print(f"  Pred stab diffs:     {pred_stability_diffs}")
    
    # Compute metrics for collapse
    metrics_collapse = compute_metrics(pred_critical_collapse, pred_stability_diffs, test_ddgs)
    # Compute metrics for crossing
    metrics_crossing = compute_metrics(pred_critical_crossing, pred_stability_diffs, test_ddgs)
    
    print("\nEvaluation Metrics (Collapse-based):")
    for k, v in metrics_collapse.items():
        print(f"  {k:22}: {v:.4f}")
        
    print("\nEvaluation Metrics (Crossing-based):")
    for k, v in metrics_crossing.items():
        print(f"  {k:22}: {v:.4f}")
        
    print("\n=========================================")
    print("ISS Phase 3 Pipeline Integration Test PASSED successfully!")
    print("=========================================")
    return True

if __name__ == "__main__":
    success = run_pipeline_integration_test()
    sys.exit(0 if success else 1)
