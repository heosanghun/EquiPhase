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
            nn.Linear(32, latent_dim),
            nn.Tanh() # Bound internally
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
        z_next = 0.9 * z + 0.1 * self.cell_net(inputs)
        return z_next
        
    def solve_fixed_point(self, X_pooled, lam, max_iter=150, tol=1e-6):
        device = X_pooled.device
        z = torch.zeros(1, self.latent_dim, device=device)
        for _ in range(max_iter):
            z_next = self.cell_forward(z, X_pooled, lam)
            if torch.norm(z_next - z, p=2, dim=-1).max() < tol:
                return z_next
            z = z_next
        return z
        
    def project_coordinates(self, z, X_proj):
        L = X_proj.shape[1]
        z_rep = z.unsqueeze(1).repeat(1, L, 1)
        z_mixed = self.mix_layer(torch.cat([z_rep, X_proj], dim=-1))
        coords = self.coord_head(z_mixed).squeeze(0)
        return coords

def compute_exact_spectral_radius(model, z_star, X_pooled, lam):
    z = z_star.clone().detach().requires_grad_(True)
    z_next = model.cell_forward(z, X_pooled, lam)
    
    D_z = z_star.shape[-1]
    J = torch.zeros(D_z, D_z, device=z_star.device)
    for i in range(D_z):
        grad = torch.autograd.grad(z_next[0, i], z, retain_graph=True)[0]
        J[i] = grad[0]
        
    eigvals = torch.linalg.eigvals(J)
    rho = torch.max(torch.abs(eigvals))
    return rho.item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
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
    
    # Use weight_decay to prevent saturation
    optimizer = optim.Adam([X_logits, aa_embed] + list(model.parameters()), lr=5e-3, weight_decay=1e-3)
    
    lam0 = torch.zeros(1, 1, device=device)
    lam1 = torch.ones(1, 1, device=device)
    
    epochs = 150
    tau_init = 1.5
    tau_min = 0.1
    
    print("Training with weight decay and leaky residual...")
    for epoch in range(1, epochs + 1):
        tau = max(tau_min, tau_init * (tau_min / tau_init) ** (epoch / epochs))
        optimizer.zero_grad()
        loss_list = []
        for idx in range(num_designs):
            y = F.gumbel_softmax(X_logits[idx:idx+1], tau=tau, hard=True, dim=-1)
            X_seq = torch.matmul(y, aa_embed)
            X_proj = model.esm_proj(X_seq)
            X_pooled = torch.mean(X_proj, dim=1)
            
            z_star_0 = model.solve_fixed_point(X_pooled, lam0)
            z_star_1 = model.solve_fixed_point(X_pooled, lam1)
            
            coords_0 = model.project_coordinates(z_star_0, X_proj)
            coords_1 = model.project_coordinates(z_star_1, X_proj)
            L_trig = torch.mean((coords_0 - Y_A)**2) + torch.mean((coords_1 - Y_B)**2)
            
            # Simple check-radius for training
            # (We can use a small power iteration)
            eps = 1e-4
            v0 = torch.ones_like(z_star_0) / np.sqrt(D_z)
            v1 = torch.ones_like(z_star_1) / np.sqrt(D_z)
            
            f_pert0 = model.cell_forward(z_star_0 + eps * v0, X_pooled, lam0)
            f_base0 = model.cell_forward(z_star_0, X_pooled, lam0)
            rho0 = torch.norm((f_pert0 - f_base0)/eps, p=2, dim=-1).mean()
            
            f_pert1 = model.cell_forward(z_star_1 + eps * v1, X_pooled, lam1)
            f_base1 = model.cell_forward(z_star_1, X_pooled, lam1)
            rho1 = torch.norm((f_pert1 - f_base1)/eps, p=2, dim=-1).mean()
            
            m_A = 1.0 - rho0
            m_B = 1.0 - rho1
            
            L_jac = torch.clamp(0.1 - m_A, min=0.0)**2 + torch.clamp(0.1 - m_B, min=0.0)**2
            L_rep = torch.clamp(4.0 - torch.norm(z_star_0 - z_star_1), min=0.0)**2
            
            loss_idx = L_trig + 10.0 * L_jac + 1.0 * L_rep
            loss_list.append(loss_idx)
            
        probs = torch.softmax(X_logits, dim=-1)
        L_div = 0.0
        for i in range(num_designs):
            for j in range(i + 1, num_designs):
                L_div += torch.mean(torch.sum(probs[i] * probs[j], dim=-1))
                
        mean_loss = torch.mean(torch.stack(loss_list))
        total_loss = mean_loss + 5.0 * L_div
        total_loss.backward()
        optimizer.step()
        
    print("Training finished. Evaluating Design 4...")
    target_idx = 3
    with torch.no_grad():
        seq_idx = torch.argmax(X_logits[target_idx], dim=-1).cpu().numpy()
        y_discrete = torch.zeros(1, L, 20, device=device)
        for pos, k in enumerate(seq_idx):
            y_discrete[0, pos, k] = 1.0
        X_seq_discrete = torch.matmul(y_discrete, aa_embed)
        X_proj = model.esm_proj(X_seq_discrete)
        X_pooled = torch.mean(X_proj, dim=1)
        
        num_sweep = 20
        sweeps = np.linspace(0.0, 1.0, num_sweep)
        
        print("\n--- Continuation Sweep with Exact margins ---")
        z = model.solve_fixed_point(X_pooled, lam0)
        for val in sweeps:
            lam = torch.tensor([[val]], dtype=torch.float32, device=device)
            for _ in range(150):
                z_next = model.cell_forward(z, X_pooled, lam)
                if torch.norm(z_next - z, p=2, dim=-1).max() < 1e-6:
                    z = z_next
                    break
                z = z_next
                
            with torch.enable_grad():
                rho = compute_exact_spectral_radius(model, z, X_pooled, lam)
            margin = 1.0 - rho
            
            coords = model.project_coordinates(z, X_proj).cpu().numpy()
            dist_A = np.sqrt(np.mean((coords - Y_A.cpu().numpy())**2))
            dist_B = np.sqrt(np.mean((coords - Y_B.cpu().numpy())**2))
            
            print(f"Lam: {val:.4f} | Margin: {margin:.4f} | distA: {dist_A:.4f} | distB: {dist_B:.4f}")

if __name__ == "__main__":
    main()
