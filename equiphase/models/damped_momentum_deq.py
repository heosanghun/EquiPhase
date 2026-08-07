import torch
import torch.nn as nn
from iss_module import ImplicitStabilitySpectroscopy

class DampedMomentumDEQ(ImplicitStabilitySpectroscopy):
    """
    Damped Momentum Deep Equilibrium Model (Heavy-Ball DEQ).
    Uses leapfrog momentum dynamics with physical friction/damping:
    - Constant damping: damping = 0.2
    - Momentum-gated damping: gated_damping = True, gamma_0 = 0.20, p_thresh = 0.1, k = 50.0
      Normalized sigmoid gate guarantees eff_damping(0) = gamma_0 = 0.20 exactly:
      gate(||p||) = sigma(k * (p_thresh - ||p||)) / sigma(k * p_thresh)
    """
    def __init__(self, esm_dim=1280, latent_dim=64, num_starts=2, dt=0.05, damping=0.2, gated_damping=False, gamma_0=0.20, p_thresh=0.1, k=50.0):
        super().__init__(esm_dim=esm_dim, latent_dim=latent_dim, num_starts=num_starts)
        self.dt = dt
        self.damping = damping
        self.gated_damping = gated_damping
        self.gamma_0 = gamma_0
        self.p_thresh = p_thresh
        self.k = k
        
        half_dim = latent_dim // 2
        
        # Gradient of potential V(q; x) network: maps [q, lam_emb] -> grad_q V
        self.grad_V_net = nn.Sequential(
            nn.Linear(half_dim + latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, half_dim)
        )
        
        # Mass matrix inverse diagonal network: maps sequence embedding to positive mass inverse
        self.mass_net = nn.Linear(latent_dim, half_dim)
        
    def cell_forward(self, z, X_pooled, lam_eff, X_mut=None, X_wt_res=None):
        """
        Leapfrog Integration Step with Damping:
        1. p_{k+1/2} = p_k - dt/2 * grad_q V(q_k; x)
        2. q_{k+1} = q_k + dt * M(x)^-1 * p_{k+1/2}
        3. p_{k+1} = (1 - damping_effective) * (p_{half} - dt/2 * grad_q V(q_{k+1}; x))
        """
        half_dim = self.latent_dim // 2
        q = z[:, :half_dim]
        p = z[:, half_dim:]
        
        # Project lam_eff to high-dimensional embedding
        lam_emb = self.lam_proj(lam_eff) # (N, latent_dim)
        
        # Compute mass inverse diagonal (positive and bounded)
        M_inv = torch.sigmoid(self.mass_net(X_pooled)) * 2.0 + 0.1 # (N, half_dim)
        
        # 1. First half-step for momentum
        grad_V_q = self.grad_V_net(torch.cat([q, lam_emb], dim=-1)) # (N, half_dim)
        p_half = p - (self.dt / 2.0) * grad_V_q
        
        # 2. Full-step for position
        q_next = q + self.dt * M_inv * p_half
        
        # 3. Second half-step for momentum
        grad_V_q_next = self.grad_V_net(torch.cat([q_next, lam_emb], dim=-1))
        p_uncut = p_half - (self.dt / 2.0) * grad_V_q_next
        
        # Determine damping factor: constant or normalized momentum-gated
        if self.gated_damping:
            p_norm = torch.norm(p_uncut, p=2, dim=-1, keepdim=True) # (N, 1)
            sigma_val = torch.sigmoid(self.k * (self.p_thresh - p_norm))
            sigma_max = torch.sigmoid(torch.tensor(self.k * self.p_thresh, device=z.device))
            norm_gate = sigma_val / sigma_max
            eff_damping = self.gamma_0 * norm_gate
        else:
            eff_damping = self.damping
            
        p_next = (1.0 - eff_damping) * p_uncut
        
        z_next = torch.cat([q_next, p_next], dim=-1)
        return z_next

# Alias for backward compatibility
SymplecticDEQ = DampedMomentumDEQ
