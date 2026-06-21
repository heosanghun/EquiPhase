import torch
import torch.nn.functional as F

def kabsch_rmsd(coords1, coords2):
    """
    Computes the optimal RMSD between coords1 and coords2 after Kabsch alignment.
    coords1, coords2: (L, 3) PyTorch tensors
    """
    device = coords1.device
    L = coords1.shape[0]
    
    # 1. Centroid alignment
    c1 = coords1.mean(dim=0)
    c2 = coords2.mean(dim=0)
    p = coords1 - c1
    q = coords2 - c2
    
    # 2. Covariance matrix
    cov = torch.matmul(p.t(), q)
    
    # 3. SVD
    try:
        u, s, v = torch.svd(cov)
    except Exception:
        # Fallback to direct RMSD if SVD fails to converge (rare)
        return torch.sqrt(torch.mean(torch.sum((p - q)**2, dim=-1)))
        
    # 4. Sign correction for right-handed coordinate system
    d = torch.det(u) * torch.det(v)
    if d < 0:
        v_corr = v.clone()
        v_corr[:, -1] = -v[:, -1]
        r = torch.matmul(u, v_corr.t())
    else:
        r = torch.matmul(u, v.t())
        
    # 5. Rotate coords1
    p_rotated = torch.matmul(p, r)
    
    # 6. Compute RMSD
    rmsd = torch.sqrt(torch.mean(torch.sum((p_rotated - q)**2, dim=-1)))
    return rmsd

def generate_matched_rmsd_decoy(X, target_rmsd, kernel_size=5, sigma=2.0, max_iter=50, tol=1e-4):
    """
    Generates a decoy structure that matches target_rmsd exactly but violates physical
    constraints via low-frequency random spatial perturbations.
    """
    device = X.device
    L = X.shape[0]
    
    # 1. Generate random noise
    noise = torch.randn_like(X)
    
    # 2. Create Gaussian smoothing filter along sequence dimension
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-x**2 / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    
    # Apply 1D conv smoothing to noise
    noise_t = noise.t().unsqueeze(0) # (1, 3, L)
    kernel_conv = kernel.view(1, 1, -1).repeat(3, 1, 1) # (3, 1, K_size)
    
    padding = kernel_size // 2
    smooth_noise_t = F.conv1d(noise_t, kernel_conv, padding=padding, groups=3)
    smooth_noise = smooth_noise_t.squeeze(0).t() # (L, 3)
    
    # Adjust length if conv padding yields dimension mismatch
    if smooth_noise.shape[0] > L:
        smooth_noise = smooth_noise[:L]
    elif smooth_noise.shape[0] < L:
        smooth_noise = torch.cat([smooth_noise, torch.zeros(L - smooth_noise.shape[0], 3, device=device)], dim=0)
        
    smooth_noise = smooth_noise - smooth_noise.mean(dim=0)
    
    # 3. Binary Search for scale factor s
    low = 0.0
    high = 10.0
    
    # Exponential search to find upper bound
    for _ in range(10):
        decoy_high = X + high * smooth_noise
        r_high = kabsch_rmsd(decoy_high, X)
        if r_high >= target_rmsd:
            break
        high *= 2.0
        
    s = (low + high) / 2.0
    for _ in range(max_iter):
        decoy = X + s * smooth_noise
        current_rmsd = kabsch_rmsd(decoy, X)
        
        if torch.abs(current_rmsd - target_rmsd) < tol:
            break
            
        if current_rmsd < target_rmsd:
            low = s
        else:
            high = s
        s = (low + high) / 2.0
        
    final_decoy = X + s * smooth_noise
    return final_decoy
