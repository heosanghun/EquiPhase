import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json

class ToySwitchDEQ(nn.Module):
    """
    A toy conditional DEQ model representing protein folding physics
    conditioned on a trigger parameter (lambda_metal).
    """
    def __init__(self, latent_dim=16, esm_proj_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        
        # ESM projection head (maps sequence to latent features)
        self.esm_proj = nn.Linear(esm_proj_dim, latent_dim)
        
        # Trigger parameter projector
        self.lam_proj = nn.Linear(1, latent_dim, bias=False)
        
        # Transition cell MLP (with residual scaling to ensure stability)
        self.cell_net = nn.Sequential(
            nn.Linear(latent_dim * 3, 32),
            nn.GELU(),
            nn.Linear(32, latent_dim)
        )
        
        # Non-linear mixing layer
        self.mix_layer = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Coordinate projection head
        self.coord_head = nn.Linear(latent_dim, 3)
        
    def cell_forward(self, z, X_pooled, lam):
        # z: [1, D_z], X_pooled: [1, D_z], lam: [1, 1]
        lam_emb = self.lam_proj(lam)
        inputs = torch.cat([z, X_pooled, lam_emb], dim=-1)
        # Bounded residual step for stability
        z_next = z + 0.1 * torch.tanh(self.cell_net(inputs))
        return z_next
        
    def solve_fixed_point(self, X_pooled, lam, max_iter=80, tol=1e-5):
        # Start from zero vector
        device = X_pooled.device
        z = torch.zeros(1, self.latent_dim, device=device)
        
        # Fixed point iteration with BPTT (differentiable)
        for _ in range(max_iter):
            z_next = self.cell_forward(z, X_pooled, lam)
            # We can check convergence but we keep gradient flow
            z = z_next
        return z
        
    def project_coordinates(self, z, X_proj):
        # z: [1, D_z], X_proj: [1, L, D_z]
        L = X_proj.shape[1]
        z_rep = z.unsqueeze(1).repeat(1, L, 1) # [1, L, D_z]
        z_mixed = self.mix_layer(torch.cat([z_rep, X_proj], dim=-1))
        coords = self.coord_head(z_mixed).squeeze(0) # [L, 3]
        return coords

def compute_spectral_radius(model, z_star, X_pooled, lam, num_iters=10):
    eps = 1e-4
    device = z_star.device
    v = torch.randn_like(z_star)
    v = v / torch.norm(v, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    
    # Power iteration to find dominant eigenvalue of Jacobian
    for _ in range(num_iters):
        f_perturbed = model.cell_forward(z_star + eps * v, X_pooled, lam)
        f_base = model.cell_forward(z_star, X_pooled, lam)
        w = (f_perturbed - f_base) / eps
        v = w / torch.norm(w, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        
    f_perturbed = model.cell_forward(z_star + eps * v, X_pooled, lam)
    f_base = model.cell_forward(z_star, X_pooled, lam)
    w = (f_perturbed - f_base) / eps
    rho = torch.norm(w, p=2, dim=-1).mean()
    return rho

def main():
    print("==========================================================")
    print("  ISS Phase 3: Inverse Jacobian switch Design PoC         ")
    print("==========================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Target Conformations Setup (Length L = 20)
    L = 20
    # Fold A: Straight Line (Consecutive CA-CA distance ~3.8 Å)
    Y_A = torch.zeros(L, 3, device=device)
    for i in range(L):
        Y_A[i, 0] = i * 3.8
        
    # Fold B: Circle (Circumference ~ L * 3.8 Å)
    Y_B = torch.zeros(L, 3, device=device)
    R = (L * 3.8) / (2 * np.pi)
    for i in range(L):
        theta = 2 * np.pi * i / L
        Y_B[i, 0] = R * np.cos(theta)
        Y_B[i, 1] = R * np.sin(theta)
        
    # 2. Initialize learnable sequence embedding
    D_z = 16
    # X_seq represents the learnable sequence representation
    X_seq = torch.randn(1, L, D_z, device=device) * 0.1
    X_seq.requires_grad = True
    
    # 3. Instantiate model
    model = ToySwitchDEQ(latent_dim=D_z, esm_proj_dim=D_z).to(device)
    
    # Optimize both embedding and model parameters to construct transition physics
    optimizer = optim.Adam([X_seq] + list(model.parameters()), lr=1e-3)
    
    # Anchoring targets
    lam0 = torch.zeros(1, 1, device=device)
    lam1 = torch.ones(1, 1, device=device)
    
    print("\n--- Training Switch Logic (100 Epochs) ---")
    
    for epoch in range(1, 101):
        optimizer.zero_grad()
        
        # Project sequence embedding
        X_proj = model.esm_proj(X_seq)
        X_pooled = torch.mean(X_proj, dim=1)
        
        # 1. State Anchoring (L_trigger)
        z_star_0 = model.solve_fixed_point(X_pooled, lam0)
        z_star_1 = model.solve_fixed_point(X_pooled, lam1)
        
        coords_0 = model.project_coordinates(z_star_0, X_proj)
        coords_1 = model.project_coordinates(z_star_1, X_proj)
        
        L_trigger_A = torch.mean((coords_0 - Y_A)**2)
        L_trigger_B = torch.mean((coords_1 - Y_B)**2)
        L_trigger = L_trigger_A + L_trigger_B
        
        # 2. Inverse Jacobian Constraint (L_jacobian)
        # Compute spectral radius of Jacobian at both states
        rho_0 = compute_spectral_radius(model, z_star_0, X_pooled, lam0)
        rho_1 = compute_spectral_radius(model, z_star_1, X_pooled, lam1)
        
        m_A = 1.0 - rho_0
        m_B = 1.0 - rho_1
        
        # Hinge loss to push margins above 0.1 (ensure stable states)
        L_jacobian = torch.clamp(0.1 - m_A, min=0.0)**2 + torch.clamp(0.1 - m_B, min=0.0)**2
        
        # 3. State Separation Repulsive Loss (L_repulsive)
        L_repulsive = torch.clamp(4.0 - torch.norm(z_star_0 - z_star_1), min=0.0)**2
        
        # Total Loss
        loss = L_trigger + 10.0 * L_jacobian + 1.0 * L_repulsive
        
        loss.backward()
        optimizer.step()
        
        # Check gradients
        grad_norm = X_seq.grad.norm().item()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} (Trig: {L_trigger.item():.4f}, Jac: {L_jacobian.item():.4f}, Rep: {L_repulsive.item():.4f}) | Grad: {grad_norm:.4e} | Margins: mA={m_A.item():.4f}, mB={m_B.item():.4f}")
            
    # Verify optimization results
    print("\n--- Verification: Sweep Trigger Lambda [0.0, 1.0] ---")
    
    # Sweep lambda
    sweeps = np.linspace(0.0, 1.0, 20)
    sweep_results = []
    
    with torch.no_grad():
        X_proj = model.esm_proj(X_seq)
        X_pooled = torch.mean(X_proj, dim=1)
        
        for val in sweeps:
            lam = torch.tensor([[val]], dtype=torch.float32, device=device)
            z_star = model.solve_fixed_point(X_pooled, lam)
            rho = compute_spectral_radius(model, z_star, X_pooled, lam)
            margin = 1.0 - rho
            
            coords = model.project_coordinates(z_star, X_proj).cpu().numpy()
            
            # Compute distance map and compare similarity to Y_A and Y_B
            # (Or simply print the margin and coordinates L2 distance)
            dist_A = float(np.sqrt(np.mean((coords - Y_A.cpu().numpy())**2)))
            dist_B = float(np.sqrt(np.mean((coords - Y_B.cpu().numpy())**2)))
            
            print(f"  Lambda: {val:.4f} | Margin: {margin.item():.4f} | Dist to A: {dist_A:.4f} | Dist to B: {dist_B:.4f}")
            sweep_results.append({
                "lambda": float(val),
                "margin": float(margin.item()),
                "dist_A": dist_A,
                "dist_B": dist_B
            })
            
    # Save sweep results for walkthrough verification
    with open("data/phase3_poc_results.json", "w") as f:
        json.dump(sweep_results, f, indent=2)
        
    print("\nVerification completed. Results saved to data/phase3_poc_results.json.")

if __name__ == "__main__":
    main()
