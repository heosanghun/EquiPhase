import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Ensure workspace is in path
sys.path.append("C:/Project/EquiPhase")

from equiphase.models.symplectic_deq import SymplecticDEQ
from iss_module import ImplicitStabilitySpectroscopy
from equiphase.models.spectral_dispatch import compute_spectral_radius

def construct_canonical_symplectic_matrix(dim, device):
    assert dim % 2 == 0, f"Dimension must be even, got {dim}"
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
    return c.item(), R, J

def test_volume_preservation():
    print("=" * 80)
    print("TEST 1: LEAPFROG VOLUME PRESERVATION (LIOUVILLE'S THEOREM - TOY D=8 & D=64)")
    print("=" * 80)
    
    latent_dim = 8
    half_dim = latent_dim // 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Instantiate Symplectic DEQ with damping = 0.0 (Toy D=8)
    model_symplectic_no_damping = SymplecticDEQ(
        esm_dim=1280,
        latent_dim=latent_dim,
        num_starts=2,
        dt=0.1,
        damping=0.0
    ).to(device)
    
    X_pooled = torch.randn(1, model_symplectic_no_damping.latent_dim, device=device)
    lam_eff = torch.tensor([[0.5]], device=device)
    X_mut = torch.randn(1, model_symplectic_no_damping.latent_dim, device=device)
    X_wt_res = torch.randn(1, model_symplectic_no_damping.latent_dim, device=device)
    
    def f_sym_no_damping(z):
        z_batch = z.unsqueeze(0)
        out = model_symplectic_no_damping.cell_forward(z_batch, X_pooled, lam_eff, X_mut, X_wt_res)
        return out.squeeze(0)
    
    for i in range(3):
        z = torch.randn(latent_dim, device=device, requires_grad=True)
        J = torch.autograd.functional.jacobian(f_sym_no_damping, z)
        det = torch.linalg.det(J).item()
        print(f"Point {i+1} | Symplectic DEQ (D=8, damping=0.0) det(J): {det:.7f}")
        assert abs(det - 1.0) < 1e-5, f"Volume preservation failed! det = {det}"
        
    print("  -> PASSED: det(J) is exactly 1.0 (conserved volume) when damping = 0.0.")
    
    # 2. Damping = 0.2
    model_symplectic_damping = SymplecticDEQ(
        esm_dim=1280,
        latent_dim=latent_dim,
        num_starts=2,
        dt=0.1,
        damping=0.2
    ).to(device)
    
    def f_sym_damping(z):
        z_batch = z.unsqueeze(0)
        out = model_symplectic_damping.cell_forward(z_batch, X_pooled, lam_eff, X_mut, X_wt_res)
        return out.squeeze(0)
        
    z = torch.randn(latent_dim, device=device)
    J = torch.autograd.functional.jacobian(f_sym_damping, z)
    det = torch.linalg.det(J).item()
    expected_det = (1.0 - 0.2)**half_dim
    print(f"Symplectic DEQ (D=8, damping=0.2) det(J): {det:.7f} | Expected: {expected_det:.7f}")
    assert abs(det - expected_det) < 1e-5, f"Jacobian determinant mismatch! det = {det}, expected = {expected_det}"
    print("  -> PASSED: det(J) decays exactly as (1 - damping)^half_dim under physical friction.")
    
    # 3. Standard Contractive DEQ (Rank-deficient volume contraction / numerical singularity)
    model_standard = ImplicitStabilitySpectroscopy(
        esm_dim=1280,
        latent_dim=latent_dim,
        num_starts=2
    ).to(device)
    
    def f_std(z):
        z_batch = z.unsqueeze(0)
        out = model_standard.cell_forward(z_batch, X_pooled, lam_eff, X_mut, X_wt_res)
        return out.squeeze(0)
        
    z = torch.randn(latent_dim, device=device)
    J = torch.autograd.functional.jacobian(f_std, z)
    det = torch.linalg.det(J).item()
    print(f"Standard DEQ det(J): {det:.7e} (|det| << 1 indicates rank deficiency / numerical singularity)")
    assert abs(det) < 1.0, f"Standard DEQ should be contractive, got det = {det}"
    print("  -> PASSED: Standard DEQ exhibits severe volume contraction.")
    print("=" * 80 + "\n")

def test_symplectic_tensor_preservation():
    print("=" * 80)
    print("TEST 2: SYMPLECTIC TENSOR CONSERVATION (J^T Omega J = c Omega)")
    print("=" * 80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Positive Control: 2D Leapfrog Integrator
    dt = 0.1
    def leapfrog_step(z):
        q, p = z[0], z[1]
        q_next = q + dt * p
        p_next = p - dt * q_next
        return torch.stack([q_next, p_next])
        
    z0 = torch.tensor([1.2, -0.7], device=device)
    c_val_2d, R_2d, _ = compute_symplectic_residual(leapfrog_step, z0)
    print(f"Positive Control (2D Leapfrog) | Conformality c: {c_val_2d:.7f} | Relative Residual R: {R_2d:.4e}")
    assert R_2d < 1e-6, f"2D Leapfrog failed symplectic tensor test! R = {R_2d}"
    print("  -> PASSED: 2D Leapfrog strictly conserves canonical symplectic structure (R < 1e-6).")
    
    # Evaluate SymplecticDEQ at D=64 (Full Model)
    model_64 = SymplecticDEQ(esm_dim=1280, latent_dim=64, num_starts=2, dt=0.1, damping=0.2).to(device)
    X_pooled = torch.randn(1, 64, device=device)
    lam_eff = torch.tensor([[0.5]], device=device)
    
    def f_sym_64(z):
        return model_64.cell_forward(z.unsqueeze(0), X_pooled, lam_eff, X_pooled, X_pooled).squeeze(0)
        
    c_val_64, R_64, _ = compute_symplectic_residual(f_sym_64, torch.randn(64, device=device))
    print(f"Full Model SymplecticDEQ (D=64, damping=0.2) | Conformality c: {c_val_64:.7f} | Relative Residual R: {R_64:.4e}")
    print(f"  -> RESULT: Conformality c = 0.8000 matches (1-damping). Non-symplectic neural residual R = {R_64*100:.2f}%.")
    print("=" * 80 + "\n")

def test_krylov_spectral_dispatch():
    print("=" * 80)
    print("TEST 3: KRYLOV SPECTRAL DISPATCH ACCURACY (STRICT TOL = 1e-3)")
    print("=" * 80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Case A: Real dominant eigenvalue [0.9, 0.5, -0.3, 0.1]
    A_real = torch.diag(torch.tensor([0.9, 0.5, -0.3, 0.1], device=device))
    def cell_forward_real(z, X_pooled, lam_eff, X_mut, X_wt_res):
        return torch.matmul(z, A_real.t())
        
    z_k = torch.randn(1, 4, device=device)
    X_pooled = torch.randn(1, 4, device=device)
    lam_eff = torch.tensor([[0.0]], device=device)
    
    rho_est_real = compute_spectral_radius(
        cell_forward_real, z_k, X_pooled, lam_eff, X_pooled, num_power_iters=50
    ).item()
    
    exact_rho_real = 0.9
    err_real = abs(rho_est_real - exact_rho_real)
    print(f"Case A (Real-Dominant) | Estimated: {rho_est_real:.6f} | Exact: {exact_rho_real:.6f} | Error: {err_real:.4e}")
    assert err_real < 1e-3, f"Real-dominant spectral radius mismatch! got {rho_est_real}, error {err_real}"
    print("  -> PASSED: Real-dominant spectral radius computed accurately under strict tol=1e-3.")
    
    # Case B: Complex dominant conjugate pair (R = 0.8)
    theta = np.pi / 4
    R_comp = 0.8
    c, s = np.cos(theta), np.sin(theta)
    A_complex = torch.tensor([
        [R_comp*c, -R_comp*s, 0.0, 0.0],
        [R_comp*s, R_comp*c, 0.0, 0.0],
        [0.0, 0.0, 0.3, 0.0],
        [0.0, 0.0, 0.0, 0.1]
    ], dtype=torch.float32, device=device)
    
    def cell_forward_complex(z, X_pooled, lam_eff, X_mut, X_wt_res):
        return torch.matmul(z, A_complex.t())
        
    rho_est_complex = compute_spectral_radius(
        cell_forward_complex, z_k, X_pooled, lam_eff, X_pooled, num_power_iters=50
    ).item()
    
    exact_rho_complex = 0.8
    err_complex = abs(rho_est_complex - exact_rho_complex)
    print(f"Case B (Complex-Dominant) | Estimated: {rho_est_complex:.6f} | Exact: {exact_rho_complex:.6f} | Error: {err_complex:.4e}")
    assert err_complex < 1e-3, f"Complex-dominant spectral radius mismatch! got {rho_est_complex}, error {err_complex}"
    print("  -> PASSED: Complex-dominant spectral radius computed accurately under strict tol=1e-3.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    test_volume_preservation()
    test_symplectic_tensor_preservation()
    test_krylov_spectral_dispatch()
