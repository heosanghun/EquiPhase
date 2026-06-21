import sys
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

# Ensure workspace is in path
sys.path.append("D:/AI/EquiPhase")

from run_real_masterpiece import train_and_evaluate_model
from iss_data import FoldSwitchDataset, split_dataset_by_family, collate_fn, parse_pdb

def test_dry_run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dry run device: {device}")
    
    # Load mutations dataset
    df = pd.read_csv("data/mutations.csv")
    pdb_ids = df["pdb_id"].tolist()
    sequences = df["sequence"].tolist()
    delta_ddgs = df["delta_ddg"].tolist()
    fold_family_ids = df["fold_family_id"].tolist()
    
    # Take 5 families for a successful split (150 samples)
    selected_indices = [i for i, f in enumerate(fold_family_ids) if f in ['fam_0', 'fam_1', 'fam_2', 'fam_3', 'fam_4']]
    pdb_ids = [pdb_ids[i] for i in selected_indices]
    sequences = [sequences[i] for i in selected_indices]
    delta_ddgs = [delta_ddgs[i] for i in selected_indices]
    fold_family_ids = [fold_family_ids[i] for i in selected_indices]
    
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
    
    print(f"Train families: {train_subset.family_ids}, Val families: {val_subset.family_ids}")
    
    import builtins
    original_range = builtins.range
    
    def patched_range(*args):
        if len(args) == 2 and args[0] == 1 and args[1] == 51:
            return original_range(1, 2)  # Stop after 1 epoch
        return original_range(*args)
        
    builtins.range = patched_range
    
    try:
        # Evaluate Standard DEQ
        train_and_evaluate_model("Standard DEQ (Baseline)", dataset, train_subset, val_subset, device)
        # Evaluate Symplectic DEQ
        train_and_evaluate_model("Symplectic DEQ (Ours)", dataset, train_subset, val_subset, device)
        print("Dry run completed successfully with no errors!")
    finally:
        builtins.range = original_range

if __name__ == "__main__":
    test_dry_run()
