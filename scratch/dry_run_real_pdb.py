import os
import sys
import pandas as pd
import torch

# Ensure workspace is in path
sys.path.append("D:/AI/EquiPhase")

from iss_data import FoldSwitchDataset, parse_pdb

def test_load():
    csv_path = "data/mutations.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    pdb_ids = df["pdb_id"].tolist()
    sequences = df["sequence"].tolist()
    delta_ddgs = df["delta_ddg"].tolist()
    fold_family_ids = df["fold_family_id"].tolist()
    
    print("Parsing PDB files...")
    target_structures_A = []
    target_structures_B = []
    
    for i, pdb_id in enumerate(pdb_ids):
        pdb_A_path = f"data/pdbs/{pdb_id}_A.pdb"
        pdb_B_path = f"data/pdbs/{pdb_id}_B.pdb"
        
        seq_A, coords_A = parse_pdb(pdb_A_path)
        seq_B, coords_B = parse_pdb(pdb_B_path)
        
        if coords_A is None or coords_B is None:
            print(f"Failed to parse PDB coordinates for {pdb_id}")
            continue
            
        target_structures_A.append(torch.tensor(coords_A, dtype=torch.float32))
        target_structures_B.append(torch.tensor(coords_B, dtype=torch.float32))
        
    print(f"Successfully parsed {len(target_structures_A)} PDB file pairs!")
    
    dataset = FoldSwitchDataset(
        sequences=sequences,
        control_params=delta_ddgs,
        delta_ddgs=delta_ddgs,
        fold_family_ids=fold_family_ids,
        pdb_ids=pdb_ids,
        esm_dim=1280
    )
    
    # Overwrite targets with parsed coordinates
    dataset.target_structures_A = target_structures_A
    dataset.target_structures_B = target_structures_B
    
    print("Dataset initialized with real coordinates successfully!")
    print(f"Sample 0 Sequence Length: {len(dataset.sequences[0])}")
    print(f"Sample 0 Target A Coordinate shape: {dataset.target_structures_A[0].shape}")
    print(f"Sample 0 Target B Coordinate shape: {dataset.target_structures_B[0].shape}")

if __name__ == "__main__":
    test_load()
