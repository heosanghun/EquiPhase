import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Ensure workspace is in path
sys.path.append("D:/AI/EquiPhase")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define physical potential V(q, S, lam)
class SyntheticPolymerPhysics:
    def __init__(self, N=10, sigma=2.0):
        self.N = N
        self.sigma = sigma
        self.k_b = 50.0  # Spring constant
        self.d_0 = 1.0   # Equilibrium bond distance
        
        # Interaction energy matrices E^A and E^B for 4 residue types (0, 1, 2, 3)
        # Type 0, 1: hydrophobic-like attraction under state A (lam=0)
        # Type 2, 3: hydrophobic-like attraction under state B (lam=1)
        self.E_A = torch.tensor([
            [-3.0, -3.0,  0.5,  0.5],
            [-3.0, -3.0,  0.5,  0.5],
            [ 0.5,  0.5,  1.0,  1.0],
            [ 0.5,  0.5,  1.0,  1.0]
        ], dtype=torch.float32)
        
        self.E_B = torch.tensor([
            [ 1.0,  1.0,  0.5,  0.5],
            [ 1.0,  1.0,  0.5,  0.5],
            [ 0.5,  0.5, -3.0, -3.0],
            [ 0.5,  0.5, -3.0, -3.0]
        ], dtype=torch.float32)

    def energy(self, q, S, lam):
        """
        q: (N, 3) - coordinates
        S: (N,) - integer sequence tokens (0 to 3)
        lam: float - control parameter
        """
        # 1. Bond energy
        diffs_bond = q[1:] - q[:-1]
        dists_bond = torch.sqrt(torch.sum(diffs_bond**2, dim=-1) + 1e-8)
        E_bond = torch.sum(self.k_b * (dists_bond - self.d_0)**2)
        
        # 2. Non-bonded interaction energy
        # Compute pairwise distances
        diffs_nonbond = q.unsqueeze(1) - q.unsqueeze(0)  # (N, N, 3)
        dists_nonbond = torch.sqrt(torch.sum(diffs_nonbond**2, dim=-1) + 1e-8)  # (N, N)
        
        # Compute sequence interaction energy E_ij
        S_i = S.unsqueeze(1).expand(-1, self.N)
        S_j = S.unsqueeze(0).expand(self.N, -1)
        
        # Gather energies
        E_ij_A = self.E_A[S_i, S_j]
        E_ij_B = self.E_B[S_i, S_j]
        E_ij = (1.0 - lam) * E_ij_A + lam * E_ij_B
        
        # Apply Gaussian RBF gate (excluding self and adjacent bonds)
        mask = torch.triu(torch.ones(self.N, self.N, device=q.device), diagonal=2)
        phi = torch.exp(-dists_nonbond**2 / (2.0 * self.sigma**2))
        
        E_nonbond = torch.sum(mask * E_ij * phi)
        return E_bond + E_nonbond

    def minimize_structure(self, S, lam, num_restarts=2):
        """
        Minimize energy w.r.t coordinates q using L-BFGS.
        """
        best_q = None
        best_energy = float('inf')
        
        for _ in range(num_restarts):
            # Initialize random chain
            q = torch.randn(self.N, 3) * 0.5
            # Make sure chain is sequential in space initially
            for i in range(1, self.N):
                q[i] = q[i-1] + torch.randn(3) * 0.2
            
            q = q.clone().detach().requires_grad_(True)
            optimizer = torch.optim.LBFGS([q], lr=0.1, max_iter=80, tolerance_grad=1e-5)
            
            def closure():
                optimizer.zero_grad()
                loss = self.energy(q, S, lam)
                loss.backward()
                return loss
                
            optimizer.step(closure)
            
            final_energy = self.energy(q, S, lam).item()
            if final_energy < best_energy and not np.isnan(final_energy):
                best_energy = final_energy
                best_q = q.clone().detach()
                
        return best_q, best_energy

def main():
    print("Generating synthetic biophysical simulation dataset...", flush=True)
    physics = SyntheticPolymerPhysics(N=10)
    
    dataset = []
    num_samples = 600  # Generate 600 sequences (sufficient for validation and robust training)
    
    for idx in range(num_samples):
        # Generate random sequence of length 10 from 4 types
        S = torch.randint(0, 4, (10,))
        
        # Minimize structure for lam=0 (Fold A)
        q_A, E_A = physics.minimize_structure(S, lam=0.0)
        # Minimize structure for lam=1 (Fold B)
        q_B, E_B = physics.minimize_structure(S, lam=1.0)
        
        # Label: which state has lower energy?
        # y = 1 if Fold A is more stable (E_A < E_B), else 0
        y = 1 if E_A < E_B else 0
        
        # Stability margin proxy from ground truth: delta energy
        delta_E = E_B - E_A
        
        dataset.append({
            "seq": S,
            "q_A": q_A,
            "q_B": q_B,
            "E_A": E_A,
            "E_B": E_B,
            "y": y,
            "delta_E": delta_E
        })
        
        if (idx + 1) % 50 == 0:
            print(f"Generated {idx + 1}/{num_samples} samples...", flush=True)
            
    # Split into disjoint sequence sets to prevent sequence leakage
    print("Constructing sequence-family disjoint splits...", flush=True)
    family_groups = {}
    for idx, item in enumerate(dataset):
        seq = item["seq"].tolist()
        family_key = tuple(np.bincount(seq, minlength=4).tolist())
        if family_key not in family_groups:
            family_groups[family_key] = []
        family_groups[family_key].append(item)
        
    print(f"Total unique sequence families: {len(family_groups)}", flush=True)
    
    keys = list(family_groups.keys())
    np.random.shuffle(keys)
    
    train_data, val_data, test_data = [], [], []
    train_split, val_split = int(len(keys) * 0.7), int(len(keys) * 0.85)
    
    for i, key in enumerate(keys):
        family_items = family_groups[key]
        if i < train_split:
            train_data.extend(family_items)
        elif i < val_split:
            val_data.extend(family_items)
        else:
            test_data.extend(family_items)
            
    print(f"Split sizes | Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}", flush=True)
    
    # Save dataset
    save_path = "D:/AI/EquiPhase/data/synthetic_physics_dataset.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "train": train_data,
        "val": val_data,
        "test": test_data
    }, save_path)
    print(f"Dataset successfully saved to {save_path}", flush=True)

if __name__ == "__main__":
    main()
