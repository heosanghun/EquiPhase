import torch

def compute_spectral_radius(cell_forward, z_k, X_pooled, lam_eff, X_mut, X_wt_res=None, num_power_iters=10):
    if X_wt_res is None:
        X_wt_res = X_pooled
        
    eps = 1e-4
    device = z_k.device
    
    # Pre-compute cell forward output at the fixed point
    fz = cell_forward(z_k, X_pooled, lam_eff, X_mut, X_wt_res)
    
    # 1. Run standard real power iteration to project the random vector into the 2D dominant subspace
    v = torch.randn_like(z_k)
    v = v / torch.norm(v, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    
    for _ in range(num_power_iters - 2):
        w = (cell_forward(z_k + eps * v, X_pooled, lam_eff, X_mut, X_wt_res) - fz) / eps
        v = w / torch.norm(w, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        
    # 2. Extract final three vectors: v0, v1 = J*v0, v2 = J*v1
    v0 = v
    v1 = (cell_forward(z_k + eps * v0, X_pooled, lam_eff, X_mut, X_wt_res) - fz) / eps
    v2 = (cell_forward(z_k + eps * v1, X_pooled, lam_eff, X_mut, X_wt_res) - fz) / eps
    
    # 3. Solver Dispatch:
    if z_k.shape[-1] == 1:
        rho = v1.abs().squeeze(-1)
    else:
        # Check collinearity for each batch item: cos(theta) = |v1.v0| / (||v1|| * ||v0||)
        d11 = torch.sum(v1 * v1, dim=-1)
        d10 = torch.sum(v1 * v0, dim=-1)
        d00 = torch.sum(v0 * v0, dim=-1)
        
        cos_theta = d10.abs() / torch.sqrt(torch.clamp(d11 * d00, min=0.0) + 1e-8).clamp(min=1e-8)
        real_dominant_mask = (cos_theta > 0.99)
        
        # If collinear (cos_theta > 0.99), the dominant eigenvalue is real, and the norm ratio is exact
        rho_real = torch.sqrt(torch.clamp(d11 / d00.clamp(min=1e-8), min=0.0) + 1e-8)
        
        # Otherwise, use 2-step subspace iteration (Krylov projection)
        d12 = torch.sum(v1 * v2, dim=-1)
        d02 = torch.sum(v0 * v2, dim=-1)
        
        det = d11 * d00 - d10**2
        
        # Safe det to prevent division by zero or large value overflow in the unused branch of torch.where
        safe_det = torch.where(real_dominant_mask, torch.ones_like(det), det)
        safe_det = torch.where(safe_det.abs() < 1e-8, torch.where(safe_det >= 0, 1e-8, -1e-8), safe_det)
        
        # Cramer's rule:
        c1 = (d02 * d10 - d12 * d00) / safe_det
        c2 = (d12 * d10 - d11 * d02) / safe_det
        
        discriminant = c1**2 - 4.0 * c2
        real_roots_mask = (discriminant >= 0)
        
        # Extract roots of x^2 + c1 x + c2 = 0
        disc_sqrt = torch.sqrt(torch.clamp(discriminant, min=0.0) + 1e-8)
        r1 = (-c1 + disc_sqrt) / 2.0
        r2 = (-c1 - disc_sqrt) / 2.0
        rho_subspace_real = torch.max(r1.abs(), r2.abs())
        
        rho_subspace_complex = torch.sqrt(torch.clamp(c2.abs(), min=0.0) + 1e-8)
        rho_subspace = torch.where(real_roots_mask, rho_subspace_real, rho_subspace_complex)
        
        # Vectorized dispatch
        rho = torch.where(real_dominant_mask, rho_real, rho_subspace)
        
    return rho
