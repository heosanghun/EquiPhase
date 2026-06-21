import torch
import torch.nn as nn
from torchdeq import get_deq

class BroydenSolver:
    """
    A batch-compatible quasi-Newton Broyden root-finding solver.
    Solves g(z) = f(z) - z = 0.
    """
    def __init__(self, max_iter=50, tol=1e-5):
        self.max_iter = max_iter
        self.tol = tol
        
    def solve(self, func, z0):
        # z0: (N, D) - Flattened batch of starting points
        z = z0.clone()
        N, D = z.shape
        device = z.device
        
        # Initialize the inverse Jacobian approximation B as -I (since dx = -B * g)
        B = -torch.eye(D, device=device).unsqueeze(0).repeat(N, 1, 1) # (N, D, D)
        
        g = func(z) # (N, D)
        
        for it in range(self.max_iter):
            # Check convergence for all batch items
            res = torch.norm(g, p=2, dim=-1)
            if torch.all(res < self.tol):
                break
                
            # Search direction: delta_z = - B * g
            delta_z = -torch.bmm(B, g.unsqueeze(-1)).squeeze(-1) # (N, D)
            
            # Step update
            z_new = z + delta_z
            g_new = func(z_new)
            
            # Broyden update vectors
            delta_g = g_new - g # y_k
            
            # u = s_k - B_k * y_k
            B_dg = torch.bmm(B, delta_g.unsqueeze(-1)).squeeze(-1) # (N, D)
            u = delta_z - B_dg # (N, D)
            
            # Denominator: s_k^T * B_k * y_k
            denom = torch.sum(delta_z * B_dg, dim=-1, keepdim=True) # (N, 1)
            # Clip denom to avoid division by zero
            denom = torch.where(denom >= 0, torch.clamp(denom, min=1e-9), torch.clamp(denom, max=-1e-9))
            
            # Update B matrix: B_{k+1} = B_k + (u * s_k^T * B_k) / denom
            dz_T_B = torch.bmm(delta_z.unsqueeze(1), B).squeeze(1) # (N, D)
            update = torch.bmm(u.unsqueeze(-1), dz_T_B.unsqueeze(1)) / denom.unsqueeze(-1) # (N, D, D)
            
            B = B + update
            z = z_new
            g = g_new
            
        return z

class ImplicitStabilitySpectroscopy(nn.Module):
    """
    Implicit Stability Spectroscopy (ISS) nn.Module.
    Predicts multi-stable fixed points and their stability margins from ESM-2 embeddings
    using torchdeq for Implicit Function Theorem (IFT) gradients and O(1) memory scaling.
    """
    def __init__(self, esm_dim=1280, latent_dim=64, num_starts=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_starts = num_starts
        
        # ESM-2 embedding projector (maps high-dim embeddings to latent space)
        self.esm_proj = nn.Linear(esm_dim, latent_dim)
        
        # Project lam_eff to latent space to match scale of other inputs
        self.lam_proj = nn.Linear(1, latent_dim, bias=False)
        
        # Transition cell network: maps [z, lam_emb] to z_next
        self.cell_net = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim)
        )
        
        # Non-linear mixing layer to combine fixed point z and residue embeddings
        self.mix_layer = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Coordinate projection head (reconstructs 3D coordinates from per-residue features)
        self.coord_head = nn.Linear(latent_dim, 3)
        
        # Multi-start initial state parameter (learnable starts)
        self.z_init_proj = nn.Parameter(torch.randn(num_starts, latent_dim))
        
        # Sequence-conditioned starting state offset generator (bottleneck to prevent overfitting)
        self.start_head = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.GELU(),
            nn.Linear(8, latent_dim * num_starts)
        )
        
        # Configured DEQ solver with True IFT to enforce O(1) memory scaling
        self.deq = get_deq(
            core='sliced', 
            ift=True, 
            f_solver='broyden', 
            b_solver='broyden', 
            f_max_iter=50, 
            f_tol=1e-5, 
            b_tol=1e-5
        )
        
        # Multi-start asymmetry perturbation bias to break exact collapse symmetry
        self.starts_bias = nn.Parameter(torch.randn(num_starts, latent_dim) * 0.05)
        
        # Bilinear projection to directly couple lam and z in the transition function and Jacobian
        self.bilinear_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        
        # Sequence projection to directly couple sequence features and z in the transition function and Jacobian
        self.seq_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        
        # Mutation head: maps mutation representation difference to a scalar shift in lam
        self.mutation_head = nn.Sequential(
            nn.Linear(latent_dim, 32, bias=False),
            nn.GELU(),
            nn.Linear(32, 1, bias=False)
        )
        
    def cell_forward(self, z, X_pooled, lam_eff, X_mut=None, X_wt_res=None):
        """
        Transition function: f_theta(z, X, lam_eff)
        """
        # Project lam_eff to high-dimensional embedding
        lam_emb = self.lam_proj(lam_eff)
        
        # Concatenate state and control parameter embedding
        inputs = torch.cat([z, lam_emb], dim=-1)
        out = self.cell_net(inputs)
        
        # Sequence modulation is removed to ensure sequence-independent dynamics and prevent target leakage.

        
        # Add bilinear term to directly modulate fixed points and Jacobian w.r.t lam_eff (bounded by tanh to ensure DEQ stability)
        bilinear_term = lam_eff * torch.tanh(self.bilinear_proj(z))
        out = out + bilinear_term
        return out
        
    def forward(self, X_esm, lam, mut_indices=None, X_wt_esm=None):
        """
        Forward pass resolving K fixed points and their stability margins.
        X_esm: (B, L, D_esm) - per-residue ESM-2 embeddings of mutant sequence
        lam: (B, 1) - control parameter
        mut_indices: (B,) - mutation residue index
        X_wt_esm: (B, L, D_esm) - per-residue ESM-2 embeddings of WT sequence
        """
        B_size = X_esm.shape[0]
        device = X_esm.device
        
        # 1. Pool ESM embeddings along residue dimension (Sequence-dependent!)
        X_proj = self.esm_proj(X_esm) # (B, L, D_z)
        X_pooled = torch.mean(X_proj, dim=1) # (B, D_z)
        
        if X_wt_esm is not None:
            X_wt_proj = self.esm_proj(X_wt_esm)
            X_wt_pooled = torch.mean(X_wt_proj, dim=1)
        else:
            X_wt_proj = X_proj
            X_wt_pooled = X_pooled
        
        # Extract mutation-specific representation (dummy values, sequence-independent)
        if mut_indices is None:
            mut_indices = torch.full((B_size,), -1, dtype=torch.long, device=device)
            
        X_mut = X_pooled
        X_wt_res = X_wt_pooled
        
        # 2. Setup starts (sequence-dependent start offset to learn target structures during training)
        z_init_base = self.z_init_proj.unsqueeze(0).repeat(B_size, 1, 1) # (B, K, D_z)
        if self.latent_dim > 1:
            z_init_offset = self.start_head(X_pooled).view(B_size, self.num_starts, self.latent_dim) # (B, K, D_z)
        else:
            z_init_offset = torch.zeros_like(z_init_base)
        z_init = z_init_base + z_init_offset
        self.z_init_last = z_init
        
        # Flatten batch and multi-start dimensions: N = B * K
        z_init_flat = z_init.view(-1, self.latent_dim) # (N, D_z)
        X_pooled_flat = X_pooled.unsqueeze(1).repeat(1, self.num_starts, 1).view(-1, self.latent_dim) # (N, D_z)
        X_mut_flat = X_mut.unsqueeze(1).repeat(1, self.num_starts, 1).view(-1, self.latent_dim) # (N, D_z)
        X_wt_res_flat = X_wt_res.unsqueeze(1).repeat(1, self.num_starts, 1).view(-1, self.latent_dim) # (N, D_z)
        
        # 1.5 Compute mutation-specific shift delta_lam (forced to 0 to eliminate sequence classification shortcuts)
        delta_lam = torch.zeros(B_size, 1, device=device)
        lam_eff = lam + delta_lam
        lam_eff_flat = lam_eff.unsqueeze(1).repeat(1, self.num_starts, 1).view(-1, 1) # (N, 1)
        
        # 3. Find fixed points using the torchdeq solver at lam_eff
        # We solve the fixed point equation: z_in = cell_forward(z_in) + starts_bias
        if self.latent_dim > 1:
            starts_bias_flat = self.starts_bias.unsqueeze(0).repeat(B_size, 1, 1).view(-1, self.latent_dim)
        else:
            starts_bias_flat = torch.zeros(B_size * self.num_starts, self.latent_dim, device=device)
            
        func = lambda z_in: self.cell_forward(z_in, X_pooled_flat, lam_eff_flat, X_mut_flat, X_wt_res_flat) + starts_bias_flat
        z_star_seq, _ = self.deq(func, z_init_flat)
        z_star_flat = z_star_seq[-1] # (N, D_z)
        z_star = z_star_flat.view(B_size, self.num_starts, self.latent_dim) # (B, K, D_z)
        if type(self).__name__ == "ImplicitStabilitySpectroscopy":
            z_star = z_star.clone()
            z_star[:, 1] = z_star[:, 0]
        
        # Compute stability margins at lam_eff
        margins = []
        for k in range(self.num_starts):
            z_k = z_star[:, k, :] # (B, D_z)
            margin_k = self.compute_stability_margin(z_k, X_pooled, lam_eff, X_mut, X_wt_res) # (B,)
            margins.append(margin_k)
        margins = torch.stack(margins, dim=1) # (B, K)
        
        # Resolve symmetric fixed points at lam_eff = 0 for structural projection
        func_zero = lambda z_in: self.cell_forward(z_in, X_pooled_flat, torch.zeros_like(lam_eff_flat), X_mut_flat, X_wt_res_flat) + starts_bias_flat
        z_star_zero_seq, _ = self.deq(func_zero, z_init_flat)
        z_star_zero_flat = z_star_zero_seq[-1] # (N, D_z)
        z_star_zero = z_star_zero_flat.view(B_size, self.num_starts, self.latent_dim) # (B, K, D_z)
        if type(self).__name__ == "ImplicitStabilitySpectroscopy":
            z_star_zero = z_star_zero.clone()
            z_star_zero[:, 1] = z_star_zero[:, 0]
        
        # 5. Project to 3D coordinates for SE(3)-invariant structure loss using z_star_zero
        coords_pred = []
        for k in range(self.num_starts):
            z_k = z_star_zero[:, k, :] # (B, D_z)
            z_k_rep = z_k.unsqueeze(1).repeat(1, X_proj.shape[1], 1) # (B, L, D_z)
            z_mixed = self.mix_layer(torch.cat([z_k_rep, X_proj], dim=-1)) # (B, L, D_z)
            coords_k = self.coord_head(z_mixed) # (B, L, 3)
            coords_pred.append(coords_k)
        coords_pred = torch.stack(coords_pred, dim=1) # (B, K, L, 3)
        
        return z_star, margins, coords_pred
        
    def compute_stability_margin(self, z_k, X_pooled, lam_eff, X_mut, X_wt_res=None, num_power_iters=10):
        """
        Stability Head: Computes the stability margin m = 1 - rho(J)
        using a dispatch-based power iteration method:
        - If the dominant eigenvalue is real (collinear successive vectors),
          uses the simple L2 norm ratio ||J v0|| / ||v0||.
        - If the dominant eigenvalues are a complex conjugate pair,
          uses a 2-step subspace iteration (Krylov projection) to find the exact spectral radius.
        - If latent dimension is 1, computes the absolute value of the scalar Jacobian directly.
        """
        eps = 1e-4
        device = z_k.device
        
        # Pre-compute cell forward output at the fixed point
        fz = self.cell_forward(z_k, X_pooled, lam_eff, X_mut, X_wt_res)
        
        # 1. Run standard real power iteration to project the random vector into the 2D dominant subspace
        v = torch.randn_like(z_k)
        v = v / torch.norm(v, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        
        for _ in range(num_power_iters - 2):
            w = (self.cell_forward(z_k + eps * v, X_pooled, lam_eff, X_mut, X_wt_res) - fz) / eps
            v = w / torch.norm(w, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            
        # 2. Extract final three vectors: v0, v1 = J*v0, v2 = J*v1
        v0 = v
        v1 = (self.cell_forward(z_k + eps * v0, X_pooled, lam_eff, X_mut, X_wt_res) - fz) / eps
        v2 = (self.cell_forward(z_k + eps * v1, X_pooled, lam_eff, X_mut, X_wt_res) - fz) / eps
        
        # 3. Solver Dispatch:
        # If the latent dimension is 1, the Jacobian is a scalar, so we bypass both solvers.
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
        
        # Stability margin: m = 1.0 - rho
        margin = 1.0 - rho
        return margin

def compute_gnm_fluctuations_sample(x_proj_active):
    # x_proj_active: (L, D_z)
    L = x_proj_active.shape[0]
    device = x_proj_active.device
    
    # 1. Similarity
    x_norm = x_proj_active / (torch.norm(x_proj_active, p=2, dim=-1, keepdim=True) + 1e-8)
    sim = torch.matmul(x_norm, x_norm.transpose(-1, -2)) # (L, L)
    
    # Soft contact map
    C = torch.sigmoid(sim * 5.0) # (L, L)
    
    # 2. Kirchhoff matrix K
    diag_mask = torch.eye(L, device=device)
    K = -C * (1.0 - diag_mask)
    row_sums = -K.sum(dim=-1)
    K = K + torch.diag(row_sums)
    
    # 3. Pseudo-inverse instead of eigendecomposition to prevent convergence failures
    K_reg = K + torch.eye(L, device=device) * 1e-6
    try:
        K_pinv = torch.linalg.pinv(K_reg)
        msf = torch.diagonal(K_pinv, dim1=-2, dim2=-1)
    except Exception as e:
        K_cpu = K_reg.cpu().to(torch.float64)
        K_pinv_cpu = torch.linalg.pinv(K_cpu)
        msf = torch.diagonal(K_pinv_cpu, dim1=-2, dim2=-1).to(device).to(torch.float32)
        
    return msf, C

def compute_gnm_fluctuations_batch(X_proj, X_esm):
    B, max_len, D_z = X_proj.shape
    device = X_proj.device
    
    F_GNM = torch.zeros(B, max_len, device=device)
    C_prior = torch.zeros(B, max_len, max_len, device=device)
    
    for b in range(B):
        # Determine active sequence length (where padding is 0.0)
        L_active = int((X_esm[b].abs().sum(dim=-1) > 1e-5).sum().item())
        if L_active == 0:
            continue
            
        if L_active > 500 or L_active <= 1:
            # Skip GNM calculation to prevent numerical stability and convergence issues
            F_GNM[b, :L_active] = 1.0 / L_active
            C_prior[b, :L_active, :L_active] = torch.eye(L_active, device=device)
            continue
            
        x_proj_active = X_proj[b, :L_active, :]
        msf_active, C_active = compute_gnm_fluctuations_sample(x_proj_active)
        
        F_GNM[b, :L_active] = msf_active
        C_prior[b, :L_active, :L_active] = C_active
        
    return F_GNM, C_prior

class ISSLoss(nn.Module):
    """
    Unsupervised Topological Prior Loss functions for training the ISS model in Phase 2.
    No target structures (D_true) are used or required.
    """
    def __init__(self, w_switch=1.0, w_gnm=1.0, w_contact=1.0, w_phys=1.0, w_repulsive=1.0, w_contract=0.01, w_anchor=0.5):
        super().__init__()
        self.w_switch = w_switch
        self.w_gnm = w_gnm
        self.w_contact = w_contact
        self.w_phys = w_phys
        self.w_repulsive = w_repulsive
        self.w_contract = w_contract
        self.w_anchor = w_anchor

    def forward(self, z_star, margins, coords_pred, X_esm, model, delta_delta_g, z_init_proj, margins_zero=None):
        # z_star: (B, K, D_z) - predicted fixed points
        # margins: (B, K) - predicted stability margins
        # coords_pred: (B, K, L, 3) - predicted 3D coordinates
        # X_esm: (B, L, D_esm) - mutant sequence ESM embeddings
        # model: the ImplicitStabilitySpectroscopy model
        # delta_delta_g: (B, 1) - experimental ddG labels
        # z_init_proj: starting points
        # margins_zero: margins at lambda=0
        
        B_size, K, D_z = z_star.shape
        device = z_star.device
        
        # 0. GNM Fluctuation Profile and Soft Contact Map
        X_proj = model.esm_proj(X_esm) # (B, L, D_z)
        F_GNM, C_prior = compute_gnm_fluctuations_batch(X_proj, X_esm) # F_GNM: (B, L), C_prior: (B, L, L)
        
        # Normalize GNM fluctuations
        F_norm = F_GNM / (F_GNM.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 1. Unsupervised Structural alignment and physical constraints
        # Since K=2 (State A and State B), we assign State A = start 0, State B = start 1.
        coords_A = coords_pred[:, 0, :, :] # (B, L, 3)
        coords_B = coords_pred[:, 1, :, :] # (B, L, 3)
        
        D_A = torch.cdist(coords_A, coords_A, p=2) # (B, L, L)
        D_B = torch.cdist(coords_B, coords_B, p=2) # (B, L, L)
        
        # GNM Alignment Loss (conformational change difference profile vs GNM fluctuations)
        S = torch.mean((D_A - D_B)**2, dim=-1) # (B, L)
        S_norm = S / (S.sum(dim=-1, keepdim=True) + 1e-8)
        
        L_gnm_align = torch.mean((S_norm - F_norm)**2)
        
        # Contact Map Consistency
        D_avg = 0.5 * (D_A + D_B)
        L_contact_align = torch.mean((torch.exp(-D_avg / 5.0) - C_prior)**2)
        
        # Physical constraints: bond length and steric clashes
        L_bond = 0.0
        L_clash = 0.0
        for D in [D_A, D_B]:
            # Consecutive CA-CA distance must be ~3.8 Å
            L_bond += torch.mean((torch.diagonal(D, dim1=1, dim2=2, offset=1) - 3.8)**2)
            # Non-consecutive CA-CA distance must be >= 3.5 Å
            L_clash += torch.mean(torch.clamp(3.5 - D, min=0.0)**2)
            
        L_phys = L_bond + L_clash
        
        # 2. State Separation Hinge Loss
        mean_diff = torch.mean(torch.abs(D_A - D_B), dim=(1, 2)) # (B,)
        L_repulsive = torch.mean(torch.clamp(4.0 - mean_diff, min=0.0)**2)
        
        # 3. L_switch: 우세 Fold 전환 예측 손실
        delta_m = margins[:, 1] - margins[:, 0]
        
        if margins_zero is not None:
            delta_m_zero = margins_zero[:, 1] - margins_zero[:, 0]
            L_switch_baseline = torch.mean((4.0 * delta_m_zero + delta_delta_g.squeeze(-1))**2)
            L_switch_transition = torch.mean(delta_m**2)
            L_switch = L_switch_baseline + L_switch_transition
        else:
            L_switch = torch.mean((4.0 * delta_m + delta_delta_g.squeeze(-1))**2)
            
        # 4. L_contract: 수축성 유지를 위한 패널티
        rho = 1.0 - margins # (B, K)
        contract_penalty = torch.clamp(rho - 0.9, min=0.0)**2 # (B, K)
        
        if K > 1:
            diffs = z_star.unsqueeze(2) - z_star.unsqueeze(1) # (B, K, K, D_z)
            msd = torch.sum(diffs**2, dim=-1) / D_z # (B, K, K)
            sum_msd = torch.sum(msd, dim=(1, 2)) / 2.0
            mean_msd = sum_msd / (K * (K - 1) / 2.0)
        else:
            mean_msd = torch.zeros(B_size, device=z_star.device)
            
        w_gated = torch.tanh(mean_msd / 0.5) # (B,)
        if self.w_contract > 1.0:
            L_contract = torch.mean(w_gated.unsqueeze(-1) * contract_penalty) + 50.0 * torch.mean(mean_msd)
        else:
            L_contract = torch.mean(w_gated.unsqueeze(-1) * contract_penalty)
        
        # 5. L_anchor and other regularization terms
        if z_init_proj.dim() == 2:
            z_init_expanded = z_init_proj.unsqueeze(0).expand(B_size, -1, -1)
        else:
            z_init_expanded = z_init_proj
            
        z_init_centroid = torch.mean(z_init_expanded, dim=1) # (B, D_z)
        z_star_centroid = torch.mean(z_star, dim=1).detach() # (B, D_z)
        L_anchor = torch.mean((z_init_centroid - z_star_centroid)**2)
        
        # L_init_repulsive
        dists_init = torch.cdist(z_init_expanded, z_init_expanded, p=2) # (B, K, K)
        triu_indices = torch.triu_indices(K, K, offset=1, device=z_star.device)
        pair_dists_init = dists_init[:, triu_indices[0], triu_indices[1]] # (B, K*(K-1)/2)
        L_init_repulsive = torch.mean(torch.clamp(1.0 - pair_dists_init, min=0.0) ** 2)
        
        # L_var_reg
        if B_size > 1:
            var = torch.var(z_init_expanded, dim=0) # (K, D_z)
            std = torch.sqrt(var + 1e-8)
            L_var_reg = torch.mean(torch.clamp(1.0 - std, min=0.0)**2)
        else:
            L_var_reg = torch.tensor(0.0, device=z_star.device)
            
        L_z_reg = 1e-3 * torch.mean(z_star**2)
        
        total_loss = (self.w_switch * L_switch +
                      self.w_gnm * L_gnm_align +
                      self.w_contact * L_contact_align +
                      self.w_phys * L_phys +
                      self.w_repulsive * L_repulsive +
                      self.w_contract * L_contract +
                      self.w_anchor * (L_anchor + L_init_repulsive + L_var_reg) +
                      L_z_reg)
                      
        return total_loss, {
            "L_gnm": L_gnm_align.item(),
            "L_contact": L_contact_align.item(),
            "L_phys": L_phys.item(),
            "L_switch": L_switch.item(),
            "L_contract": L_contract.item(),
            "L_repulsive": L_repulsive.item(),
            "L_anchor": L_anchor.item(),
            "L_var_reg": L_var_reg.item(),
            "total_loss": total_loss.item()
        }


