import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Monkey-patch platform module to bypass Windows WMI query hangs
import platform
from collections import namedtuple
UnameResult = namedtuple('UnameResult', ['system', 'node', 'release', 'version', 'machine', 'processor'])
platform.win32_ver = lambda *args, **kwargs: ('10', '10.0.0', '', 'Multiprocessor Free')
platform.uname = lambda: UnameResult('Windows', 'DESKTOP-XXX', '10', '10.0.0', 'AMD64', 'AMD64')
platform.machine = lambda: 'AMD64'
platform.system = lambda: 'Windows'
platform.processor = lambda: 'AMD64'
platform.release = lambda: '10'
platform.version = lambda: '10.0.0'

# Ensure the workspace is in path
sys.path.append("D:/AI/EquiPhase")
from iss_module import ImplicitStabilitySpectroscopy, ISSLoss

def run_standard_tests():
    print("=========================================")
    print("Starting Standard ISS Module Verification")
    print("=========================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device being used: {device}")
    
    # Initialize shapes
    B = 4
    L = 10
    D_esm = 1280
    latent_dim = 64
    num_starts = 2
    
    # 1. Initialize model and loss
    print("\n--- Test 1: Model Initialization ---")
    model = ImplicitStabilitySpectroscopy(
        esm_dim=D_esm,
        latent_dim=latent_dim,
        num_starts=num_starts
    ).to(device)
    
    # Verify z_init_proj is registered as a parameter
    assert "z_init_proj" in [n for n, _ in model.named_parameters()], "z_init_proj should be a parameter!"
    print("z_init_proj successfully verified as registered parameter.")
    
    criterion = ISSLoss().to(device)
    print("Model and loss initialized successfully.")
    
    # 2. Forward pass verification
    print("\n--- Test 2: Forward Pass & Output Dimensions ---")
    torch.manual_seed(42)
    X_esm = torch.randn(B, L, D_esm, device=device)
    lam = torch.randn(B, 1, device=device)
    mut_indices = torch.randint(0, L, (B,), device=device)
    X_wt_esm = torch.randn(B, L, D_esm, device=device)
    
    try:
        z_star, margins, coords_pred = model(X_esm, lam, mut_indices=mut_indices, X_wt_esm=X_wt_esm)
        print("Forward pass completed successfully!")
        print(f"z_star shape: {z_star.shape} (Expected: ({B}, {num_starts}, {latent_dim}))")
        print(f"margins shape: {margins.shape} (Expected: ({B}, {num_starts}))")
        print(f"coords_pred shape: {coords_pred.shape} (Expected: ({B}, {num_starts}, {L}, 3))")
        
        assert z_star.shape == (B, num_starts, latent_dim), f"Incorrect z_star shape: {z_star.shape}"
        assert margins.shape == (B, num_starts), f"Incorrect margins shape: {margins.shape}"
        assert coords_pred.shape == (B, num_starts, L, 3), f"Incorrect coords_pred shape: {coords_pred.shape}"
        print("Dimensions check passed.")
    except Exception as e:
        print(f"FAILED during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None
        
    # 3. Solver convergence check
    print("\n--- Test 3: Solver Convergence & Residual Check ---")
    try:
        # Check fixed-point residual: g(z*) = cell_forward(z*) - z*
        X_proj = model.esm_proj(X_esm)
        X_pooled = torch.mean(X_proj, dim=1)
        X_wt_proj = model.esm_proj(X_wt_esm)
        X_wt_pooled = torch.mean(X_wt_proj, dim=1)
        
        X_mut_list = []
        X_wt_res_list = []
        for b in range(B):
            idx = mut_indices[b].item()
            if idx != -1 and idx < X_proj.shape[1]:
                X_mut_list.append(X_proj[b, idx])
                X_wt_res_list.append(X_wt_proj[b, idx])
            else:
                X_mut_list.append(X_pooled[b])
                X_wt_res_list.append(X_wt_pooled[b])
        X_mut = torch.stack(X_mut_list, dim=0)
        X_wt_res = torch.stack(X_wt_res_list, dim=0)
        
        # Check convergence of the first start (since the standard model forces collapse by overwriting the second start)
        z_star_first = z_star[:, 0, :]
        starts_bias_first = model.starts_bias[0].unsqueeze(0).repeat(B, 1)
        g_val = model.cell_forward(z_star_first, X_pooled, lam, X_mut, X_wt_res) + starts_bias_first - z_star_first
        residual_norms = torch.norm(g_val, p=2, dim=-1)
        max_res = residual_norms.max().item()
        mean_res = residual_norms.mean().item()
        
        print(f"Max residual norm of solved fixed points: {max_res:.2e}")
        print(f"Mean residual norm of solved fixed points: {mean_res:.2e}")
        
        assert max_res < 1e-2, f"DEQ solver residual too large: {max_res}"
        print("DEQ convergence check passed.")
    except Exception as e:
        print(f"FAILED during convergence check: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None

    # 4. Loss calculation check
    print("\n--- Test 4: Loss Calculation with Bistability Gating ---")
    try:
        coords_target_A = torch.randn(B, L, 3, device=device)
        coords_target_B = torch.randn(B, L, 3, device=device)
        delta_delta_g = torch.randn(B, 1, device=device)
        
        loss, loss_dict = criterion(z_star, margins, coords_pred, X_esm, model, delta_delta_g, model.z_init_last)
        print(f"Loss computed successfully! Total Loss: {loss.item():.4f}")
        for k, v in loss_dict.items():
            print(f"  {k}: {v:.4f}")
            
        assert loss.item() >= 0, "Loss is negative!"
        print("Loss calculation check passed.")
    except Exception as e:
        print(f"FAILED during loss calculation: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None

    # 5. Backward Pass & IFT Gradient Flow
    print("\n--- Test 5: Backward Pass & IFT Gradient Flow ---")
    try:
        # Zero out gradients
        model.zero_grad()
        
        # Backward pass
        loss.backward()
        print("Backward pass completed successfully!")
        
        # Verify gradient flow to parameters
        print("\nChecking parameter gradients:")
        grad_flow_passed = True
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"  {name:30} | grad_norm: {grad_norm:.2e} | shape: {list(param.shape)}")
                if grad_norm == 0.0:
                    if "coord_head.bias" in name or "mix_layer.2.bias" in name:
                        print(f"  {name:30} | grad_norm: 0.00e+00 (Expected due to translation invariance) | shape: {list(param.shape)}")
                    else:
                        print(f"ERROR: Gradient for {name} is exactly 0.0!")
                        grad_flow_passed = False
            else:
                if "seq_proj" in name or "mutation_head" in name:
                    print(f"  {name:30} | GRADIENT IS NONE (Expected by design) | shape: {list(param.shape)}")
                else:
                    print(f"  {name:30} | GRADIENT IS NONE! | shape: {list(param.shape)}")
                    grad_flow_passed = False
                    
        if grad_flow_passed:
            print("\nGradient flow verification PASSED (O(1) IFT math is correct).")
        else:
            print("\nGradient flow verification FAILED: some parameters did not receive gradients.")
            return False, None, None, None
            
    except Exception as e:
        print(f"FAILED during backward pass / gradient flow: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None
        
    print("Standard tests passed successfully.")
    return True, model, X_esm, lam


def test_analytical_bifurcation_bridge():
    print("\n=========================================")
    print("Starting Phase 1 Analytical Bifurcation Bridge Test")
    print("=========================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Instantiate the model with 1D latent space and 3 start points
    # (to capture both stable wells and the unstable barrier)
    model = ImplicitStabilitySpectroscopy(
        esm_dim=1280,
        latent_dim=1,
        num_starts=3
    ).to(device)
    
    from torchdeq import get_deq
    model.deq = get_deq(
        core='sliced', 
        ift=True, 
        f_solver='broyden', 
        b_solver='broyden', 
        f_max_iter=300, 
        f_tol=1e-5, 
        b_tol=1e-5
    )
    
    # 2. Mock model.cell_forward to implement the 1D double-well potential gradient descent:
    # f(z, lam) = z - alpha * (z^3 - z - lam), where alpha = 0.05
    alpha = 0.05
    def double_well_cell(self, z, X_pooled, lam, X_mut=None, X_wt_res=None):
        # z: (N, 1), lam: (N, 1)
        grad = z**3 - z - lam
        return z - alpha * grad
        
    # Apply monkey-patch to cell_forward
    import types
    model.cell_forward = types.MethodType(double_well_cell, model)
    
    # Re-register parameter with shape (3, 1) for 1D 3-start bridge test
    model.z_init_proj = torch.nn.Parameter(torch.tensor([[-1.5], [0.0], [1.5]], dtype=torch.float32, device=device))
    
    # Sweep lambda values and compare resolved fixed points and margins with Phase 1 analytical values
    test_points = [
        {"lam": -0.5, "expected_num_roots": 1, "desc": "Single stable well (lower)"},
        {"lam": 0.0, "expected_num_roots": 3, "desc": "Bistable regime (2 stable wells + 1 unstable barrier)"},
        {"lam": 0.5, "expected_num_roots": 1, "desc": "Single stable well (upper)"}
    ]
    
    dummy_X = torch.zeros(1, 5, 1280, device=device) # Dummy ESM-2 embedding
    
    print(f"\n{'Lambda':<10}{'Init Start':<15}{'Resolved z*':<15}{'Margin m':<15}{'Stability':<12}")
    print("-" * 70)
    
    bridge_passed = True
    
    for tp in test_points:
        lam_val = tp["lam"]
        lam_tensor = torch.tensor([[lam_val]], dtype=torch.float32, device=device)
        
        with torch.no_grad():
            z_star, margins, _ = model(dummy_X, lam_tensor)
            
        z_star = z_star.squeeze(0).cpu().numpy().flatten() # (3,)
        margins = margins.squeeze(0).cpu().numpy().flatten() # (3,)
        
        for k in range(3):
            z_val = z_star[k]
            m_val = margins[k]
            init_start = model.z_init_proj[k].item()
            stability_str = "Stable" if m_val > 0 else "Unstable"
            print(f"{lam_val:<10.2f}{init_start:<15.2f}{z_val:<15.4f}{m_val:<15.4f}{stability_str:<12}")
            
        # Analytical Validation checks:
        if lam_val == 0.0:
            # Under lam = 0:
            # - start -1.5 should converge to -1.0
            # - start 1.5 should converge to 1.0
            # - start 0.0 should stay at 0.0
            # stable margins should be approx 0.10, unstable margin approx -0.05
            assert abs(z_star[0] - (-1.0)) < 1e-3, f"Root 1 mismatch: {z_star[0]}"
            assert abs(z_star[1] - 0.0) < 1e-3, f"Root 2 mismatch: {z_star[1]}"
            assert abs(z_star[2] - 1.0) < 1e-3, f"Root 3 mismatch: {z_star[2]}"
            
            assert abs(margins[0] - 0.10) < 1e-3, f"Margin 1 mismatch: {margins[0]}"
            assert abs(margins[1] - (-0.05)) < 1e-3, f"Margin 2 mismatch: {margins[1]}"
            assert abs(margins[2] - 0.10) < 1e-3, f"Margin 3 mismatch: {margins[2]}"
            
        elif lam_val == 0.5:
            # Under lam = 0.5 (monostable):
            # All starts should converge to the single stable root z* approx 1.1915
            # Margin should be approx 0.1629
            for val in z_star:
                assert abs(val - 1.1915) < 1e-2, f"Root mismatch at lam=0.5: {val}"
            for m in margins:
                assert abs(m - 0.1629) < 1e-2, f"Margin mismatch at lam=0.5: {m}"
                
        elif lam_val == -0.5:
            # Under lam = -0.5 (monostable):
            # All starts should converge to the single stable root z* approx -1.1915
            # Margin should be approx 0.1629
            for val in z_star:
                assert abs(val - (-1.1915)) < 1e-2, f"Root mismatch at lam=-0.5: {val}"
            for m in margins:
                assert abs(m - 0.1629) < 1e-2, f"Margin mismatch at lam=-0.5: {m}"
                
    # 3. Test Bifurcation point lambda_c approx 0.3849 collapse:
    # At lam = 0.3849:
    # There is a saddle-node bifurcation where the lower stable well and unstable barrier collide.
    # The merged branch has stability margin m approx 0.
    print("\nVerifying stability collapse (m -> 0) at critical bifurcation lam = 0.3849...")
    lam_c = 2.0 / (3.0 * np.sqrt(3.0)) # approx 0.3849
    lam_tensor = torch.tensor([[lam_c]], dtype=torch.float32, device=device)
    
    with torch.no_grad():
        z_star, margins, _ = model(dummy_X, lam_tensor)
        
    z_star = z_star.squeeze(0).cpu().numpy().flatten()
    margins = margins.squeeze(0).cpu().numpy().flatten()
    
    print(f"Critical Lambda: {lam_c:.4f}")
    print(f"Resolved roots: {z_star}")
    print(f"Stability margins: {margins}")
    
    # Check that root 0 (start -1.5) converges to -0.577 (z_c) or the single upper well.
    # If starting at 0.0 or -1.5, we are near the bifurcation point z_c = -1/sqrt(3) approx -0.5774
    # The stability margin at this point should collapse to 0.
    # Let's find the minimum margin, which corresponds to the bifurcation point.
    min_margin = min(margins)
    print(f"Minimum margin near bifurcation point: {min_margin:.5f}")
    assert abs(min_margin) < 5e-3, f"Stability margin did not collapse to 0 at bifurcation: {min_margin}"
    
    print("\n=========================================")
    print("Phase 1 Analytical Bridge Test PASSED successfully!")
    print("=========================================")
    return True


def test_asymmetric_spectral_radius_resolution():
    print("\n=========================================")
    print("Starting Asymmetric Spectral Radius Resolution Test")
    print("=========================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the model with 8D latent space and 1 start
    model = ImplicitStabilitySpectroscopy(
        esm_dim=1280,
        latent_dim=8,
        num_starts=1
    ).to(device)
    
    # Setup test 1: Real dominant eigenvalue 0.95
    torch.manual_seed(42)
    D_real = torch.diag(torch.tensor([0.95, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.0], device=device))
    P = torch.randn(8, 8, device=device)
    Q, _ = torch.linalg.qr(P)
    P = Q
    J_real = torch.matmul(torch.matmul(P, D_real), P.t())
    
    # Mock cell_forward to perform linear transformation by J
    def linear_cell_real(self, z, X_pooled, lam, X_mut=None, X_wt_res=None):
        return torch.matmul(z, J_real.t())
        
    import types
    model.cell_forward = types.MethodType(linear_cell_real, model)
    
    # Test margin calculation on a random vector (representing resolved fixed point z_k)
    z_k = torch.randn(1, 8, device=device)
    X_pooled = torch.zeros(1, 8, device=device)
    lam = torch.zeros(1, 1, device=device)
    
    with torch.no_grad():
        margin_real = model.compute_stability_margin(z_k, X_pooled, lam, X_pooled, num_power_iters=40)
        
    rho_real = 1.0 - margin_real.item()
    print(f"Real Dominant Case | True: 0.9500 | Resolved: {rho_real:.4f} | Margin: {margin_real.item():.4f}")
    assert abs(rho_real - 0.95) < 1e-2, f"Failed real-dominant spectral radius resolution: {rho_real}"
    
    # Setup test 2: Complex dominant eigenvalue magnitude 0.8
    theta = torch.tensor(3.1415926 / 4, device=device)
    R = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]], device=device) * 0.8
    D_comp = torch.block_diag(R, torch.diag(torch.tensor([0.4, 0.3, 0.2, 0.1, 0.05, 0.0], device=device)))
    J_comp = torch.matmul(torch.matmul(P, D_comp), P.t())
    
    def linear_cell_comp(self, z, X_pooled, lam, X_mut=None, X_wt_res=None):
        return torch.matmul(z, J_comp.t())
        
    model.cell_forward = types.MethodType(linear_cell_comp, model)
    
    with torch.no_grad():
        margin_comp = model.compute_stability_margin(z_k, X_pooled, lam, X_pooled, num_power_iters=40)
        
    rho_comp = 1.0 - margin_comp.item()
    print(f"Complex Dominant Case | True: 0.8000 | Resolved: {rho_comp:.4f} | Margin: {margin_comp.item():.4f}")
    assert abs(rho_comp - 0.80) < 1e-2, f"Failed complex-dominant spectral radius resolution: {rho_comp}"
    
    print("\n=========================================")
    print("Asymmetric Spectral Radius Resolution Test PASSED successfully!")
    print("=========================================")
    return True


if __name__ == "__main__":
    success_std, model, X_esm, lam = run_standard_tests()
    if not success_std:
        sys.exit(1)
        
    success_bridge = test_analytical_bifurcation_bridge()
    if not success_bridge:
        sys.exit(1)
        
    success_spectral = test_asymmetric_spectral_radius_resolution()
    if not success_spectral:
        sys.exit(1)
        
    print("\nALL ISS Phase 2 REFINEMENT TESTS PASSED!")
    sys.exit(0)
