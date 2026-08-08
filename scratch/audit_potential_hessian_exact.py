import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append("C:/Project/EquiPhase")
from equiphase.models.potential_damped_deq import PotentialDampedDEQ

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

print("==========================================================================================")
print("=== 1. EXACT SECOND-ORDER AUTOGRAD TRACE (NO GRAPH DETACH) ===")
print("==========================================================================================")

torch.manual_seed(42)
model_pot = PotentialDampedDEQ(esm_dim=1280, latent_dim=64, num_starts=2, dt=0.1, damping=0.20).to(device)
X_pooled = torch.randn(1, 64, device=device)
lam_eff = torch.tensor([[0.5]], device=device)
lam_emb = model_pot.lam_proj(lam_eff)

def f_potential_linked(z, dt_val=0.1):
    half_dim = 32
    q = z[:half_dim].unsqueeze(0)
    p = z[half_dim:].unsqueeze(0)
    M_inv = torch.sigmoid(model_pot.mass_net(X_pooled)) * 2.0 + 0.1
    
    # Do NOT detach q! Maintain full graph trace to input z!
    inputs = torch.cat([q, lam_emb], dim=-1)
    V_val = model_pot.V_net(inputs)
    grad_V = torch.autograd.grad(outputs=V_val.sum(), inputs=q, create_graph=True, retain_graph=True)[0]
    
    p_half = p - (dt_val / 2.0) * grad_V
    q_next = q + dt_val * M_inv * p_half
    
    inputs_next = torch.cat([q_next, lam_emb], dim=-1)
    V_val_next = model_pot.V_net(inputs_next)
    grad_V_next = torch.autograd.grad(outputs=V_val_next.sum(), inputs=q_next, create_graph=True, retain_graph=True)[0]
    
    p_uncut = p_half - (dt_val / 2.0) * grad_V_next
    p_next = (1.0 - 0.20) * p_uncut
    
    return torch.cat([q_next, p_next], dim=-1).squeeze(0)

# Check Jacobian of get_conservative_force
def get_conservative_force_linked(q):
    inputs = torch.cat([q.unsqueeze(0), lam_emb], dim=-1)
    V_val = model_pot.V_net(inputs)
    grad_V = torch.autograd.grad(outputs=V_val.sum(), inputs=q, create_graph=True, retain_graph=True)[0]
    return grad_V

q_test = torch.randn(32, device=device)
J_c = torch.autograd.functional.jacobian(get_conservative_force_linked, q_test)
norm_full_c = torch.linalg.norm(J_c, ord="fro").item()
norm_diff_c = torch.linalg.norm(J_c - J_c.t(), ord="fro").item()
anti_sym_c_pct = (norm_diff_c / (norm_full_c + 1e-12)) * 100

print(f"Force Jacobian Frobenius Norm ||J_c||_F: {norm_full_c:.8f}")
print(f"Anti-Symmetric Difference ||J_c - J_c^T||_F: {norm_diff_c:.4e}")
print(f"Relative Anti-Symmetry Metric: {anti_sym_c_pct:.4e}%")

print("\n==========================================================================================")
print("=== 2. dt^2 RESIDUAL SCALING SWEEP TEST (EXACT GRAPH TRACE) ===")
print("==========================================================================================")

z_test = torch.randn(64, device=device)

for dt_v in [0.10, 0.05, 0.02, 0.01]:
    def f_wrap(z):
        return f_potential_linked(z, dt_val=dt_v)
        
    c_dt, R_dt = compute_symplectic_residual(f_wrap, z_test)
    print(f"dt = {dt_v:4.2f} | Conformality c = {c_dt:.6f} | Residual R = {R_dt*100:.6f}% ({R_dt:.6e})")
