import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import json
import random
import hashlib

# 1. Set seed for strict reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

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
    print("  ISS Phase 3.1: Gumbel-Softmax Discrete Sequence Design  ")
    print("==========================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Random Seed: {SEED}")
    
    # Define 20 standard amino acids
    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
    
    # 1. Target Conformations Setup (Length L = 20)
    L = 20
    # Fold A: Straight Line
    Y_A = torch.zeros(L, 3, device=device)
    for i in range(L):
        Y_A[i, 0] = i * 3.8
        
    # Fold B: Circle
    Y_B = torch.zeros(L, 3, device=device)
    R = (L * 3.8) / (2 * np.pi)
    for i in range(L):
        theta = 2 * np.pi * i / L
        Y_B[i, 0] = R * np.cos(theta)
        Y_B[i, 1] = R * np.sin(theta)
        
    # 2. Setup Learnable Parameters
    D_z = 16
    num_designs = 5
    
    # X_logits representing logits over 20 amino acids at each position
    X_logits = nn.Parameter(torch.randn(num_designs, L, 20, device=device) * 0.1)
    
    # Shared learnable amino acid embedding table
    aa_embed = nn.Parameter(torch.randn(20, D_z, device=device) * 0.1)
    
    # Model
    model = ToySwitchDEQ(latent_dim=D_z, esm_proj_dim=D_z).to(device)
    
    optimizer = optim.Adam([X_logits, aa_embed] + list(model.parameters()), lr=5e-3)
    
    lam0 = torch.zeros(1, 1, device=device)
    lam1 = torch.ones(1, 1, device=device)
    
    # Hyperparameters
    epochs = 150
    tau_init = 1.5
    tau_min = 0.1
    
    print("\n--- Training Discrete Switch Logic ---")
    
    for epoch in range(1, epochs + 1):
        # Anneal Gumbel-Softmax Temperature
        tau = max(tau_min, tau_init * (tau_min / tau_init) ** (epoch / epochs))
        
        optimizer.zero_grad()
        
        loss_list = []
        L_trigger_list = []
        L_jacobian_list = []
        L_repulsive_list = []
        
        # We loop over the designs to avoid modifying ToySwitchDEQ batch size logic
        for idx in range(num_designs):
            # Apply Gumbel-Softmax (hard=True for discrete-like forward, soft gradients backward)
            y = F.gumbel_softmax(X_logits[idx:idx+1], tau=tau, hard=True, dim=-1) # [1, L, 20]
            
            # Map one-hot probabilities to continuous space using the learned embeddings
            X_seq = torch.matmul(y, aa_embed) # [1, L, D_z]
            
            X_proj = model.esm_proj(X_seq)
            X_pooled = torch.mean(X_proj, dim=1)
            
            # 1. State Anchoring (L_trigger)
            z_star_0 = model.solve_fixed_point(X_pooled, lam0)
            z_star_1 = model.solve_fixed_point(X_pooled, lam1)
            
            coords_0 = model.project_coordinates(z_star_0, X_proj)
            coords_1 = model.project_coordinates(z_star_1, X_proj)
            
            L_trigger_A = torch.mean((coords_0 - Y_A)**2)
            L_trigger_B = torch.mean((coords_1 - Y_B)**2)
            L_trig = L_trigger_A + L_trigger_B
            
            # 2. Inverse Jacobian Constraint (L_jacobian)
            rho_0 = compute_spectral_radius(model, z_star_0, X_pooled, lam0)
            rho_1 = compute_spectral_radius(model, z_star_1, X_pooled, lam1)
            
            m_A = 1.0 - rho_0
            m_B = 1.0 - rho_1
            
            L_jac = torch.clamp(0.1 - m_A, min=0.0)**2 + torch.clamp(0.1 - m_B, min=0.0)**2
            
            # 3. State Separation Repulsive Loss (L_repulsive)
            L_rep = torch.clamp(4.0 - torch.norm(z_star_0 - z_star_1), min=0.0)**2
            
            # Accumulate design-specific loss
            loss_idx = L_trig + 10.0 * L_jac + 1.0 * L_rep
            loss_list.append(loss_idx)
            
            L_trigger_list.append(L_trig.item())
            L_jacobian_list.append(L_jac.item())
            L_repulsive_list.append(L_rep.item())
            
        # 4. Diversity Penalty (orthogonality in sequence logits distributions)
        probs = torch.softmax(X_logits, dim=-1) # [5, L, 20]
        L_div = 0.0
        for i in range(num_designs):
            for j in range(i + 1, num_designs):
                # Orthogonality/overlap penalty across the 5 designs
                L_div += torch.mean(torch.sum(probs[i] * probs[j], dim=-1))
                
        # Combine all components
        mean_design_loss = torch.mean(torch.stack(loss_list))
        total_loss = mean_design_loss + 5.0 * L_div
        
        total_loss.backward()
        optimizer.step()
        
        grad_norm = X_logits.grad.norm().item()
        
        if epoch % 15 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {total_loss.item():.4f} (Trig: {np.mean(L_trigger_list):.4f}, Jac: {np.mean(L_jacobian_list):.4f}, Rep: {np.mean(L_repulsive_list):.4f}, Div: {L_div.item():.4f}) | Grad: {grad_norm:.4e} | Temp: {tau:.4f}")

    print("\n--- Final Sequence Generation & Verification ---")
    
    designed_sequences = []
    results_data = []
    
    # 20 sweeps for trigger lambda to verify transitions
    sweeps = np.linspace(0.0, 1.0, 20)
    
    with torch.no_grad():
        for idx in range(num_designs):
            # Resolve discrete sequence by argmax
            seq_idx = torch.argmax(X_logits[idx], dim=-1).cpu().numpy()
            aa_seq = "".join([AMINO_ACIDS[k] for k in seq_idx])
            designed_sequences.append(aa_seq)
            
            # Form pure discrete one-hot vector (no gumbel noise)
            y_discrete = torch.zeros(1, L, 20, device=device)
            for pos, k in enumerate(seq_idx):
                y_discrete[0, pos, k] = 1.0
                
            X_seq_discrete = torch.matmul(y_discrete, aa_embed)
            X_proj = model.esm_proj(X_seq_discrete)
            X_pooled = torch.mean(X_proj, dim=1)
            
            # Evaluate margins at endpoint triggers
            z_star_A = model.solve_fixed_point(X_pooled, lam0)
            z_star_B = model.solve_fixed_point(X_pooled, lam1)
            
            rho_A = compute_spectral_radius(model, z_star_A, X_pooled, lam0)
            rho_B = compute_spectral_radius(model, z_star_B, X_pooled, lam1)
            
            mA_final = float((1.0 - rho_A).item())
            mB_final = float((1.0 - rho_B).item())
            
            # Sweep lambda for spinodal collapse behavior
            sweep_points = []
            for val in sweeps:
                lam = torch.tensor([[val]], dtype=torch.float32, device=device)
                z_star = model.solve_fixed_point(X_pooled, lam)
                rho = compute_spectral_radius(model, z_star, X_pooled, lam)
                margin = float((1.0 - rho).item())
                
                coords = model.project_coordinates(z_star, X_proj).cpu().numpy()
                dist_A = float(np.sqrt(np.mean((coords - Y_A.cpu().numpy())**2)))
                dist_B = float(np.sqrt(np.mean((coords - Y_B.cpu().numpy())**2)))
                
                sweep_points.append({
                    "lambda": float(val),
                    "margin": margin,
                    "dist_A": dist_A,
                    "dist_B": dist_B
                })
                
            results_data.append({
                "design_index": idx,
                "sequence": aa_seq,
                "margin_A": mA_final,
                "margin_B": mB_final,
                "sweep": sweep_points
            })
            
            print(f"\nDesign {idx+1}: {aa_seq}")
            print(f"  Endpoint Margins -> mA: {mA_final:.4f}, mB: {mB_final:.4f}")
            # Find spinodal collapse (minimum margin)
            min_sweep = min(sweep_points, key=lambda x: x["margin"])
            print(f"  Spinodal collapse at lambda = {min_sweep['lambda']:.4f} with margin = {min_sweep['margin']:.4f}")
            print(f"  Endpoint A dist: {sweep_points[0]['dist_A']:.4f} | Endpoint B dist: {sweep_points[-1]['dist_B']:.4f}")

    # Generate FASTA text
    fasta_lines = []
    for idx, seq in enumerate(designed_sequences):
        fasta_lines.append(f">design_{idx+1}_mA_{results_data[idx]['margin_A']:.4f}_mB_{results_data[idx]['margin_B']:.4f}")
        fasta_lines.append(seq)
    fasta_content = "\n".join(fasta_lines) + "\n"
    
    # Save results
    os.makedirs("data", exist_ok=True)
    with open("data/phase3_1_fasta.fa", "w") as f:
        f.write(fasta_content)
        
    with open("data/phase3_1_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    # Save checkpoint for plotting
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "X_logits": X_logits,
        "aa_embed": aa_embed
    }
    torch.save(checkpoint, "data/design4_checkpoint.pt")
    print("Saved model checkpoint to data/design4_checkpoint.pt")
        
    # Generate hash for verification
    code_hash = hashlib.sha256(fasta_content.encode('utf-8')).hexdigest()
    
    print("\n==========================================================")
    print("  FASTA Sequences:")
    print("==========================================================")
    print(fasta_content.strip())
    print(f"File saved: data/phase3_1_fasta.fa")
    print(f"JSON data saved: data/phase3_1_results.json")
    print(f"FASTA SHA256 Hash: {code_hash}")
    print("==========================================================")

if __name__ == "__main__":
    main()
