import torch
import torch.nn as nn
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

# 1. EquiPhase DEQ (Ours) - 32D Latent Space
class EquiPhaseDoubleWellDEQ(nn.Module):
    def __init__(self, latent_dim=32, damping=0.20, dt=0.10):
        super().__init__()
        self.latent_dim = latent_dim
        self.damping = damping
        self.dt = dt
        
        self.fc1 = nn.Linear(latent_dim, 64)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(64, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        
        A_diag = torch.tensor([1.0, 0.3] + [-0.5]*(latent_dim-2), device=device)
        self.A = torch.diag(A_diag)

    def V_total(self, q):
        q_sq = torch.sum(q**2, dim=-1, keepdim=True)
        q_A_q = torch.sum(q * torch.matmul(q, self.A.t()), dim=-1, keepdim=True)
        v_base = 0.25 * (q_sq**2) - 0.5 * q_A_q
        v_net = self.fc2(self.act(self.fc1(q)))
        return v_base + v_net
        
    def compute_force(self, q):
        q_req = q if q.requires_grad else q.detach().requires_grad_(True)
        V_val = self.V_total(q_req)
        grad_V = torch.autograd.grad(outputs=V_val.sum(), inputs=q_req, create_graph=True, retain_graph=True)[0]
        return grad_V
        
    def cell_forward(self, z, dt_v=0.10):
        q = z[:self.latent_dim].unsqueeze(0)
        p = z[self.latent_dim:].unsqueeze(0)
        
        grad_V = self.compute_force(q)
        p_half = p - (dt_v / 2.0) * grad_V
        q_next = q + dt_v * p_half
        
        grad_V_next = self.compute_force(q_next)
        p_uncut = p_half - (dt_v / 2.0) * grad_V_next
        p_next = (1.0 - self.damping) * p_uncut
        
        return torch.cat([q_next, p_next], dim=-1).squeeze(0)

# 2. Vanilla DEQ
class VanillaDEQ(nn.Module):
    def __init__(self, state_dim=64):
        super().__init__()
        self.W = nn.Linear(state_dim, state_dim)
        
    def cell_forward(self, z):
        return torch.tanh(self.W(z))

# 3. Monotone DEQ (monDEQ)
class MonotoneDEQ(nn.Module):
    def __init__(self, state_dim=64, m=0.1):
        super().__init__()
        self.M = nn.Linear(state_dim, state_dim, bias=False)
        self.m = m
        self.state_dim = state_dim
        
    def get_W(self):
        W_raw = self.M.weight
        W_sym = torch.matmul(W_raw.t(), W_raw) + self.m * torch.eye(self.state_dim, device=device)
        return -W_sym
        
    def cell_forward(self, z):
        W = self.get_W()
        return torch.relu(torch.matmul(W, z))

def run_audit():
    print("==========================================================================================")
    print("=== PAPER 2 THREE-WAY BASELINE COMPARISON: EQUIPHASE vs VANILLA DEQ vs MONDEQ ===")
    print("==========================================================================================")
    
    models = {
        "EquiPhase DEQ (Ours)": EquiPhaseDoubleWellDEQ().to(device),
        "Vanilla DEQ (Baseline 1)": VanillaDEQ().to(device),
        "Monotone DEQ (Baseline 2)": MonotoneDEQ().to(device)
    }
    
    results = {}
    
    for name, model in models.items():
        torch.manual_seed(42)
        z_test = torch.randn(64, device=device)
        f_map = lambda z: model.cell_forward(z)
        
        c_val, R_val = compute_symplectic_residual(f_map, z_test)
        
        stable_basins = []
        saddle_points = []
        diverged_count = 0
        
        for init_seed in range(100):
            torch.manual_seed(3000 + init_seed)
            z_curr = torch.randn(64, device=device) * 2.0
            
            for s in range(500):
                z_curr = f_map(z_curr)
                if torch.isnan(z_curr).any() or torch.isinf(z_curr).any() or torch.norm(z_curr) > 1e4:
                    diverged_count += 1
                    break
            else:
                J_f = torch.autograd.functional.jacobian(f_map, z_curr)
                rho_J_f = compute_spectral_radius_power_method(J_f, num_iters=30)
                
                if rho_J_f < 1.0:
                    stable_basins.append(z_curr.detach())
                else:
                    saddle_points.append(z_curr.detach())
                    
        # Count unique attractor basins
        if "EquiPhase" in name:
            q1_vals = [b[0].item() for b in stable_basins]
            plus_count = sum(1 for q in q1_vals if q > 0.1)
            minus_count = sum(1 for q in q1_vals if q < -0.1)
            unique_basins = (1 if plus_count > 0 else 0) + (1 if minus_count > 0 else 0)
        else:
            unique_basins = 1 if len(stable_basins) > 0 else 0
            
        print(f"\n[{name}]")
        print(f"  - Conformal Scale c           : {c_val:.6f}")
        print(f"  - Symplectic Violation R      : {R_val:.6e}")
        print(f"  - Diverged Trajectories       : {diverged_count}/100")
        print(f"  - Stable Attractors (rho < 1) : {len(stable_basins)} (Unique Basins = {unique_basins})")
        print(f"  - Saddle Points (rho >= 1)    : {len(saddle_points)}")
        
        results[name] = {
            "c": c_val,
            "R": R_val,
            "unique_basins": unique_basins,
            "stable_attractors": len(stable_basins)
        }
        
    return results

if __name__ == "__main__":
    run_audit()
