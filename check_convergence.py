import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

class ToySwitchDEQ(nn.Module):
    def __init__(self, latent_dim=16, esm_proj_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.esm_proj = nn.Linear(esm_proj_dim, latent_dim)
        self.lam_proj = nn.Linear(1, latent_dim, bias=False)
        self.cell_net = nn.Sequential(
            nn.Linear(latent_dim * 3, 32),
            nn.GELU(),
            nn.Linear(32, latent_dim)
        )
        self.mix_layer = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.coord_head = nn.Linear(latent_dim, 3)
        
    def cell_forward(self, z, X_pooled, lam):
        lam_emb = self.lam_proj(lam)
        inputs = torch.cat([z, X_pooled, lam_emb], dim=-1)
        # Leaky residual cell update (0.9 * z + 0.1 * tanh(...) to bound state and allow non-zero Jacobian)
        z_next = 0.9 * z + 0.1 * torch.tanh(self.cell_net(inputs))
        return z_next
        
    def project_coordinates(self, z, X_proj):
        L = X_proj.shape[1]
        z_rep = z.unsqueeze(1).repeat(1, L, 1)
        z_mixed = self.mix_layer(torch.cat([z_rep, X_proj], dim=-1))
        coords = self.coord_head(z_mixed).squeeze(0)
        return coords

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    L = 20
    Y_A = torch.zeros(L, 3, device=device)
    for i in range(L):
        Y_A[i, 0] = i * 3.8
    Y_B = torch.zeros(L, 3, device=device)
    R = (L * 3.8) / (2 * np.pi)
    for i in range(L):
        theta = 2 * np.pi * i / L
        Y_B[i, 0] = R * np.cos(theta)
        Y_B[i, 1] = R * np.sin(theta)
        
    D_z = 16
    num_designs = 5
    X_logits = nn.Parameter(torch.randn(num_designs, L, 20, device=device) * 0.1)
    aa_embed = nn.Parameter(torch.randn(20, D_z, device=device) * 0.1)
    model = ToySwitchDEQ(latent_dim=D_z, esm_proj_dim=D_z).to(device)
    optimizer = optim.Adam([X_logits, aa_embed] + list(model.parameters()), lr=5e-3)
    
    lam0 = torch.zeros(1, 1, device=device)
    lam1 = torch.ones(1, 1, device=device)
    
    print("Training with leaky residual cell update...")
    for epoch in range(1, 51):
        optimizer.zero_grad()
        loss_list = []
        for idx in range(num_designs):
            y = F.gumbel_softmax(X_logits[idx:idx+1],
                                 tau=max(0.1, 1.5 * (0.1/1.5)**(epoch/50)),
                                 hard=True, dim=-1)
            X_seq = torch.matmul(y, aa_embed)
            X_proj = model.esm_proj(X_seq)
            X_pooled = torch.mean(X_proj, dim=1)
            
            # Solve with convergence check during training
            z_0 = torch.zeros(1, D_z, device=device)
            for _ in range(80):
                z_next = model.cell_forward(z_0, X_pooled, lam0)
                if torch.norm(z_next - z_0).item() < 1e-4:
                    z_0 = z_next
                    break
                z_0 = z_next
                
            z_1 = torch.zeros(1, D_z, device=device)
            for _ in range(80):
                z_next = model.cell_forward(z_1, X_pooled, lam1)
                if torch.norm(z_next - z_1).item() < 1e-4:
                    z_1 = z_next
                    break
                z_1 = z_next
                
            coords_0 = model.project_coordinates(z_0, X_proj)
            coords_1 = model.project_coordinates(z_1, X_proj)
            L_trig = torch.mean((coords_0 - Y_A)**2) + torch.mean((coords_1 - Y_B)**2)
            loss_list.append(L_trig)
            
        total_loss = torch.mean(torch.stack(loss_list))
        total_loss.backward()
        optimizer.step()
        
    print("Training completed. Let's observe fixed point convergence at lambda=0.0:")
    with torch.no_grad():
        y = F.gumbel_softmax(X_logits[3:4], tau=0.1, hard=True, dim=-1)
        X_seq = torch.matmul(y, aa_embed)
        X_proj = model.esm_proj(X_seq)
        X_pooled = torch.mean(X_proj, dim=1)
        
        z = torch.zeros(1, D_z, device=device)
        for it in range(1, 101):
            z_next = model.cell_forward(z, X_pooled, lam0)
            diff = torch.norm(z_next - z).item()
            z = z_next
            if it <= 10 or it % 10 == 0:
                print(f"  Iter {it:03d} | z norm: {torch.norm(z).item():.6f} | residual: {diff:.2e}")
                
if __name__ == "__main__":
    main()
