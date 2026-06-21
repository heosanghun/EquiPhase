import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Ensure workspace is in path
sys.path.append("D:/AI/EquiPhase")

from equiphase.models.symplectic_deq import SymplecticDEQ
from iss_module import ImplicitStabilitySpectroscopy
from equiphase.models.spectral_dispatch import compute_spectral_radius

def test_volume_preservation():
    print("=" * 80)
    print("TEST 1: LEAPFROG VOLUME PRESERVATION (LIOUVILLE'S THEOREM)")
    print("=" * 80)
    
    latent_dim = 8
    half_dim = latent_dim // 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Instantiate Symplectic DEQ with damping = 0.0
    model_symplectic_no_damping = SymplecticDEQ(
        esm_dim=1280,
        latent_dim=latent_dim,
        num_starts=2,
        dt=0.1,
        damping=0.0
    ).to(device)
    
    # Generate dummy input variables
    X_pooled = torch.randn(1, model_symplectic_no_damping.latent_dim, device=device)
    lam_eff = torch.tensor([[0.5]], device=device)
    X_mut = torch.randn(1, model_symplectic_no_damping.latent_dim, device=device)
    X_wt_res = torch.randn(1, model_symplectic_no_damping.latent_dim, device=device)
    
    # Define transition function wrapper
    def f_sym_no_damping(z):
        z_batch = z.unsqueeze(0)
        out = model_symplectic_no_damping.cell_forward(z_batch, X_pooled, lam_eff, X_mut, X_wt_res)
        return out.squeeze(0)
    
    # Test at multiple random points in phase space
    for i in range(3):
        z = torch.randn(latent_dim, device=device, requires_grad=True)
        J = torch.autograd.functional.jacobian(f_sym_no_damping, z)
        det = torch.linalg.det(J).item()
        print(f"Point {i+1} | Symplectic DEQ (damping=0.0) det(J): {det:.7f}")
        assert abs(det - 1.0) < 1e-5, f"Volume preservation failed! det = {det}"
        
    print("  -> PASSED: det(J) is exactly 1.0 (conserved volume) when damping = 0.0.")
    
    # 2. Instantiate Symplectic DEQ with damping = 0.2
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
    print(f"Symplectic DEQ (damping=0.2) det(J): {det:.7f} | Expected: {expected_det:.7f}")
    assert abs(det - expected_det) < 1e-5, f"Jacobian determinant mismatch! det = {det}, expected = {expected_det}"
    print("  -> PASSED: det(J) decays exactly as (1 - damping)^half_dim under physical friction.")
    
    # 3. Contrast with standard contractive DEQ
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
    print(f"Standard DEQ (Contractive) det(J):  {det:.7e} (det < 1.0 implies volume contraction)")
    assert abs(det) < 1.0, f"Standard DEQ should be contractive, got det = {det}"
    print("  -> PASSED: Standard DEQ exhibits severe volume contraction, leading to basin collapse.")
    print("=" * 80 + "\n")

def test_krylov_spectral_dispatch():
    print("=" * 80)
    print("TEST 2: KRYLOV SPECTRAL DISPATCH ACCURACY")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We will test the compute_spectral_radius function on structured linear operators A.
    # We construct a wrapper function mapping f(z) = A z.
    # The Jacobian of f(z) w.r.t z is constant and equal to A.
    # Thus, the spectral radius of J is exactly the maximum absolute eigenvalue of A.
    
    # Case A: Real dominant eigenvalue
    # eigenvalues = [0.9, 0.5, -0.3, 0.1]
    # This should trigger collinearity (cos_theta > 0.99) and run 1D power iteration.
    A_real = torch.diag(torch.tensor([0.9, 0.5, -0.3, 0.1], device=device))
    
    def cell_forward_real(z, X_pooled, lam_eff, X_mut, X_wt_res):
        return torch.matmul(z, A_real.t())
        
    z_k = torch.randn(1, 4, device=device)
    X_pooled = torch.randn(1, 4, device=device)
    lam_eff = torch.tensor([[0.0]], device=device)
    
    # Let's manually trace compute_spectral_radius logic
    rho_est_real = compute_spectral_radius(
        cell_forward_real, z_k, X_pooled, lam_eff, X_pooled, num_power_iters=50
    ).item()
    
    exact_rho_real = 0.9
    print(f"Case A (Real-Dominant) | Estimated: {rho_est_real:.6f} | Exact: {exact_rho_real:.6f}")
    assert abs(rho_est_real - exact_rho_real) < 1e-3, f"Real-dominant spectral radius mismatch! got {rho_est_real}"
    print("  -> PASSED: Real-dominant spectral radius computed accurately.")
    
    # Case B: Complex dominant conjugate pair
    # Construct block rotation scaled by R = 0.8
    # eigenvalues = 0.8 * e^(i theta) -> spectral radius = 0.8
    theta = np.pi / 4 # 45 degrees
    R = 0.8
    c, s = np.cos(theta), np.sin(theta)
    
    # 4x4 matrix with a 2x2 complex dominant rotation and 2x2 decay
    A_complex = torch.tensor([
        [R*c, -R*s, 0.0, 0.0],
        [R*s, R*c, 0.0, 0.0],
        [0.0, 0.0, 0.3, 0.0],
        [0.0, 0.0, 0.0, 0.1]
    ], dtype=torch.float32, device=device)
    
    def cell_forward_complex(z, X_pooled, lam_eff, X_mut, X_wt_res):
        return torch.matmul(z, A_complex.t())
        
    rho_est_complex = compute_spectral_radius(
        cell_forward_complex, z_k, X_pooled, lam_eff, X_pooled, num_power_iters=50
    ).item()
    
    exact_rho_complex = 0.8
    print(f"Case B (Complex-Dominant) | Estimated: {rho_est_complex:.6f} | Exact: {exact_rho_complex:.6f}")
    assert abs(rho_est_complex - exact_rho_complex) < 5e-3, f"Complex-dominant spectral radius mismatch! got {rho_est_complex}"
    print("  -> PASSED: Complex-dominant spectral radius computed accurately via 2-step Krylov projection.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    test_volume_preservation()
    test_krylov_spectral_dispatch()
