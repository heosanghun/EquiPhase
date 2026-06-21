import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys

# Ensure workspace is in path
sys.path.append("D:/AI/EquiPhase")

from equiphase.models.symplectic_deq import SymplecticDEQ
from equiphase.models.losses import MasterpieceLoss

# 1. Custom Dataset
class SyntheticPhysicsDataset(Dataset):
    def __init__(self, items):
        self.items = items
        
    def __len__(self):
        return len(self.items)
        
    def __getitem__(self, idx):
        item = self.items[idx]
        return {
            "seq": item["seq"],
            "q_A": item["q_A"],
            "q_B": item["q_B"],
            "delta_E": torch.tensor(item["delta_E"], dtype=torch.float32)
        }

# Collate function to format batches matching model expectation
def collate_fn(batch):
    seqs = torch.stack([x["seq"] for x in batch])
    q_As = torch.stack([x["q_A"] for x in batch])
    q_Bs = torch.stack([x["q_B"] for x in batch])
    delta_Es = torch.stack([x["delta_E"] for x in batch])
    return seqs, q_As, q_Bs, delta_Es

# 2. Simple Sequence Embedder
class SyntheticSeqEmbedder(nn.Module):
    def __init__(self, vocab_size=4, embed_dim=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
    def forward(self, x):
        return self.emb(x)

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on Device: {device}")
    
    # Load dataset
    data_path = "D:/AI/EquiPhase/data/synthetic_physics_dataset.pt"
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}. Please run data generation first.")
        sys.exit(1)
        
    data = torch.load(data_path)
    train_set = SyntheticPhysicsDataset(data["train"])
    val_set = SyntheticPhysicsDataset(data["val"])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Initialize networks
    embedder = SyntheticSeqEmbedder(vocab_size=4, embed_dim=128).to(device)
    model = SymplecticDEQ(
        esm_dim=128,
        latent_dim=64,
        num_starts=2,
        dt=0.05,
        damping=0.2
    ).to(device)
    
    # Apply orthogonal initialization to starting positions
    with torch.no_grad():
        torch.nn.init.orthogonal_(model.z_init_proj)
    print("Applied orthogonal initialization to z_init_proj.")
    
    # Loss criterion
    # Use MasterpieceLoss with balanced weights
    criterion = MasterpieceLoss(
        tau=0.1,
        gamma=2.0,
        w_repulsive=1.5,
        w_anchor=1.0,
        w_switch=2.0,
        w_stability=0.5
    ).to(device)
    
    # Optimize both embedder and model parameters
    optimizer = optim.Adam(
        list(embedder.parameters()) + list(model.parameters()),
        lr=1e-3,
        weight_decay=1e-5
    )
    
    epochs = 40
    best_val_loss = float('inf')
    best_weights = None
    
    print("\nStarting Symplectic MDEQ training on synthetic physics...")
    for epoch in range(1, epochs + 1):
        # Training loop
        model.train()
        embedder.train()
        train_loss = 0.0
        train_collapse_sum = 0.0
        
        for seqs, q_As, q_Bs, delta_Es in train_loader:
            seqs = seqs.to(device)
            q_As = q_As.to(device)
            q_Bs = q_Bs.to(device)
            delta_Es = delta_Es.to(device)
            
            optimizer.zero_grad()
            
            # Embed sequences
            X_esm = embedder(seqs) # (B, L, 128)
            
            # Run model forward pass
            # We train at lambda=0.5 to let the model experience intermediate transition dynamics
            # while sculpting the end-state basins at lambda=0
            lams = torch.full((seqs.shape[0], 1), 0.5, device=device)
            z_star, margins, coords_pred = model(X_esm, lams)
            
            # Compute loss
            loss, loss_dict = criterion(coords_pred, q_As, q_Bs, z_star, margins=margins, delta_delta_g=delta_Es)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(embedder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * seqs.shape[0]
            
            # Collapse rate
            dists = torch.cdist(z_star, z_star, p=2)
            max_dists = dists.max(dim=-1)[0].max(dim=-1)[0]
            train_collapse_sum += (max_dists < 1e-2).float().sum().item()
            
        train_loss /= len(train_set)
        train_collapse = train_collapse_sum / len(train_set)
        
        # Validation loop
        model.eval()
        embedder.eval()
        val_loss = 0.0
        val_collapse_sum = 0.0
        
        with torch.no_grad():
            for seqs, q_As, q_Bs, delta_Es in val_loader:
                seqs = seqs.to(device)
                q_As = q_As.to(device)
                q_Bs = q_Bs.to(device)
                delta_Es = delta_Es.to(device)
                
                X_esm = embedder(seqs)
                lams = torch.full((seqs.shape[0], 1), 0.5, device=device)
                z_star, margins, coords_pred = model(X_esm, lams)
                
                loss, loss_dict = criterion(coords_pred, q_As, q_Bs, z_star, margins=margins, delta_delta_g=delta_Es)
                val_loss += loss.item() * seqs.shape[0]
                
                dists = torch.cdist(z_star, z_star, p=2)
                max_dists = dists.max(dim=-1)[0].max(dim=-1)[0]
                val_collapse_sum += (max_dists < 1e-2).float().sum().item()
                
        val_loss /= len(val_set)
        val_collapse = val_collapse_sum / len(val_set)
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Collapse: {val_collapse:.1%}")
        
        # Save best model checkpoint (lowest Val Loss with low collapse rate)
        if val_collapse < 0.10 and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {
                "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "embedder_state_dict": {k: v.cpu().clone() for k, v in embedder.state_dict().items()}
            }
            print(f"  [Checkpoint] Saved new best model with Val Loss {val_loss:.4f}")
            
    # Save the final best weights
    save_weights_path = "D:/AI/EquiPhase/data/synthetic_symplectic_model.pt"
    if best_weights is not None:
        torch.save(best_weights, save_weights_path)
        print(f"\nBest model successfully saved to {save_weights_path}")
    else:
        # Fallback to final epoch weights
        torch.save({
            "model_state_dict": model.state_dict(),
            "embedder_state_dict": embedder.state_dict()
        }, save_weights_path)
        print(f"\nWarning: No model met the validation criteria. Saved final epoch weights to {save_weights_path}")

if __name__ == "__main__":
    train_model()
