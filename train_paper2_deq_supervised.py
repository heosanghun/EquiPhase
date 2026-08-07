import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import hashlib
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def construct_canonical_symplectic_matrix(dim, device):
    half_dim = dim // 2
    I = torch.eye(half_dim, device=device)
    Z = torch.zeros(half_dim, half_dim, device=device)
    Omega = torch.cat([
        torch.cat([Z, I], dim=1),
        torch.cat([-I, Z], dim=1)
    ], dim=0)
    return Omega

def compute_symplectic_residual(f_func, z):
    dim = z.shape[-1]
    Omega = construct_canonical_symplectic_matrix(dim, z.device)
    J = torch.autograd.functional.jacobian(f_func, z)
    
    J_T = J.t()
    J_T_Omega_J = torch.matmul(J_T, torch.matmul(Omega, J))
    c = torch.trace(torch.matmul(J_T_Omega_J, Omega.t())) / torch.trace(torch.matmul(Omega, Omega.t()))
    
    target_tensor = c * Omega
    diff_tensor = J_T_Omega_J - target_tensor
    
    res_norm = torch.linalg.norm(diff_tensor, ord="fro").item()
    target_norm = torch.linalg.norm(target_tensor, ord="fro").item()
    
    R = res_norm / (target_norm + 1e-12)
    return c.item(), R

def compute_spectral_radius_power_method(M, num_iters=30):
    v = torch.randn(M.shape[0], 1, dtype=M.dtype, device=M.device)
    v = v / torch.linalg.norm(v)
    for _ in range(num_iters):
        v = torch.matmul(M, v)
        norm_v = torch.linalg.norm(v)
        v = v / (norm_v + 1e-12)
    Mv = torch.matmul(M, v)
    rho = float(torch.abs(torch.matmul(v.t(), Mv)).item())
    return rho

# Task (c) EquiPhase Supervised DEQ Model with Sign-Paired Loss
class EquiPhaseSupervisedDEQ(nn.Module):
    def __init__(self, latent_dim=32, damping=0.20, dt=0.10):
        super().__init__()
        self.latent_dim = latent_dim
        self.damping = damping
        self.dt = dt
        
        # Neural potential V_net(q; x)
        self.fc1 = nn.Linear(latent_dim + 2, 64)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(64, 1)
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.zeros_(self.fc2.bias)
        
        # A matrix: a1 = alpha (input conditioning), a2 = 0.3 (fixed saddle), a3..32 = -0.5 (fixed harmonic)
        A_diag = torch.tensor([1.0, 0.3] + [-0.5]*(latent_dim-2), device=device)
        self.A_base = torch.diag(A_diag)

    def V_total(self, q, x):
        # q: [B, latent_dim], x: [B, 2] (x = [alpha, sqrt(alpha)])
        alpha = x[:, 0:1]
        A_dynamic = self.A_base.clone()
        
        q_sq = torch.sum(q**2, dim=-1, keepdim=True)
        # Dynamic a1 = alpha along e1
        q_A_q = alpha * (q[:, 0:1]**2) + 0.3 * (q[:, 1:2]**2) + torch.sum(-0.5 * (q[:, 2:]**2), dim=-1, keepdim=True)
        
        v_base = 0.25 * (q_sq**2) - 0.5 * q_A_q
        
        qx = torch.cat([q, x], dim=-1)
        v_net = self.fc2(self.act(self.fc1(qx)))
        return v_base + v_net
        
    def compute_force(self, q, x):
        q_req = q if q.requires_grad else q.detach().requires_grad_(True)
        V_val = self.V_total(q_req, x)
        grad_V = torch.autograd.grad(outputs=V_val.sum(), inputs=q_req, create_graph=True, retain_graph=True)[0]
        return grad_V
        
    def cell_forward_single(self, z, x_single):
        q = z[:self.latent_dim].unsqueeze(0)
        p = z[self.latent_dim:].unsqueeze(0)
        x = x_single.unsqueeze(0)
        
        grad_V = self.compute_force(q, x)
        p_half = p - (self.dt / 2.0) * grad_V
        q_next = q + self.dt * p_half
        
        grad_V_next = self.compute_force(q_next, x)
        p_uncut = p_half - (self.dt / 2.0) * grad_V_next
        p_next = (1.0 - self.damping) * p_uncut
        
        return torch.cat([q_next, p_next], dim=-1).squeeze(0)

    def solve_equilibrium(self, z_init, x, num_steps=100):
        z_curr = z_init
        for _ in range(num_steps):
            q = z_curr[:, :self.latent_dim]
            p = z_curr[:, self.latent_dim:]
            
            grad_V = self.compute_force(q, x)
            p_half = p - (self.dt / 2.0) * grad_V
            q_next = q + self.dt * p_half
            
            grad_V_next = self.compute_force(q_next, x)
            p_uncut = p_half - (self.dt / 2.0) * grad_V_next
            p_next = (1.0 - self.damping) * p_uncut
            
            z_curr = torch.cat([q_next, p_next], dim=-1)
        return z_curr

def main():
    print("==========================================================================================")
    print("=== TASK (C) SIGN-PAIRED DEQ SUPERVISED LEARNING & PREREGISTRATION AUDIT ===")
    print("==========================================================================================")
    
    model = EquiPhaseSupervisedDEQ().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Balanced 50/50 Positive and Negative Initializations
    torch.manual_seed(100)
    batch_size = 32
    half_b = batch_size // 2
    
    alphas = torch.rand(batch_size, 1, device=device) * 0.4 + 0.8  # alpha in [0.8, 1.2]
    x_batch = torch.cat([alphas, torch.sqrt(alphas)], dim=-1)     # [B, 2]
    
    # Pre-training Audit
    x_test_single = torch.tensor([1.0, 1.0], device=device)
    f_map_pre = lambda z: model.cell_forward_single(z, x_test_single)
    c_pre, R_pre = compute_symplectic_residual(f_map_pre, torch.randn(64, device=device))
    print(f"\n[Architectural Guarantee] Pre-training c: {c_pre:.6f}, Symplectic R: {R_pre:.6e}")
    
    # Execute 50 Epochs of Sign-Paired Supervised Training
    print("\n--- EXECUTING SIGN-PAIRED TASK (C) SUPERVISED DEQ TRAINING ---")
    for epoch in range(1, 51):
        optimizer.zero_grad()
        
        # 50/50 Balanced initializations: half positive q1, half negative q1
        z_init = torch.randn(batch_size, 64, device=device) * 0.5
        z_init[:half_b, 0] = torch.abs(z_init[:half_b, 0]) + 0.5   # positive half
        z_init[half_b:, 0] = -torch.abs(z_init[half_b:, 0]) - 0.5  # negative half
        
        # Sign-paired target: sign(z0_q1) * sqrt(alpha) * e1
        q0 = z_init[:, 0:1]
        target_sign = torch.sign(q0)
        target_q = torch.zeros(batch_size, 32, device=device)
        target_q[:, 0:1] = target_sign * torch.sqrt(alphas)
        
        z_star = model.solve_equilibrium(z_init, x_batch, num_steps=60)
        q_star = z_star[:, :32]
        
        loss_eq = torch.mean((q_star - target_q)**2)
        
        z_next = model.solve_equilibrium(z_star, x_batch, num_steps=1)
        loss_res = torch.mean((z_next - z_star)**2)
        
        total_loss = loss_eq + 10.0 * loss_res
        total_loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/50 | Sign-Paired Target Loss: {loss_eq.item():.6e} | Residual Penalty: {loss_res.item():.6e}")
            
    # Save Checkpoint
    checkpoint_path = "C:/Project/EquiPhase/supervised_deq_model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    
    with open(checkpoint_path, "rb") as f:
        ckpt_bytes = f.read()
        ckpt_sha256 = hashlib.sha256(ckpt_bytes).hexdigest()
        
    print(f"\nTrained Supervised Checkpoint Saved To: {checkpoint_path}")
    print(f"Checkpoint SHA-256 Hash: {ckpt_sha256}")
    
    # Post-training Audit
    print("\n--- POST-TRAINING AUDIT (Trained Model) ---")
    f_map_post = lambda z: model.cell_forward_single(z, x_test_single)
    c_post, R_post = compute_symplectic_residual(f_map_post, torch.randn(64, device=device))
    
    stable_basins = []
    saddle_points = []
    diverged_count = 0
    
    for init_seed in range(100):
        torch.manual_seed(4000 + init_seed)
        z_curr = torch.randn(64, device=device) * 2.0
        
        for s in range(300):
            z_curr = f_map_post(z_curr)
            if torch.isnan(z_curr).any() or torch.isinf(z_curr).any() or torch.norm(z_curr) > 1e4:
                diverged_count += 1
                break
        else:
            J_f = torch.autograd.functional.jacobian(f_map_post, z_curr)
            rho_J_f = compute_spectral_radius_power_method(J_f, num_iters=30)
            
            if rho_J_f < 1.0:
                stable_basins.append(z_curr.detach())
            else:
                saddle_points.append(z_curr.detach())
                
    q1_vals = [b[0].item() for b in stable_basins]
    plus_count = sum(1 for q in q1_vals if q > 0.1)
    minus_count = sum(1 for q in q1_vals if q < -0.1)
    unique_basins = (1 if plus_count > 0 else 0) + (1 if minus_count > 0 else 0)
    
    print(f"[Architectural Guarantee] Post-training c: {c_post:.6f}, Symplectic R: {R_post:.6e}")
    print(f"Diverged Trajectories: {diverged_count}/100")
    print(f"Stable Attractor Basins (rho < 1.0): {len(stable_basins)} (Plus={plus_count}, Minus={minus_count}) -> Unique Basins: {unique_basins}")
    print(f"Unstable Saddle Manifolds (rho >= 1.0): {len(saddle_points)}")

if __name__ == "__main__":
    main()
