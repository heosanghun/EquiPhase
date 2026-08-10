# -*- coding: utf-8 -*-
import hashlib, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn

DATA_PATH = os.path.join("data", "ala2", "alanine-dipeptide-3x250ns-backbone-dihedrals.npz")
SEED = 7777
SIGMA = 0.15
BATCH = 4096
LR = 1e-3
STEPS = 3000
PI = math.pi
R3_MARGIN = 0.05

def wrap(x):
    return torch.remainder(x + PI, 2 * PI) - PI

def enc(q):
    return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)

class VanillaScore(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(4, h), nn.ReLU(),
                               nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2))
    def forward(self, q):
        return self.f(enc(q))

def train_dsm_vanilla(data, seed, dev):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = VanillaScore().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    n = data.shape[0]
    for _ in range(STEPS):
        idx = torch.randint(0, n, (BATCH,), generator=g)
        q = data[idx].to(dev)
        eps = torch.randn((BATCH, 2), generator=g).to(dev) * SIGMA
        qt = wrap(q + eps)
        s = net(qt)
        loss = ((SIGMA ** 2) * s + eps).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net

def main():
    t0 = time.strftime("%Y%m%d_%H%M%S")
    print("WP2_HELMHOLTZ_BEGIN")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    npz = np.load(DATA_PATH)
    X = np.concatenate([np.asarray(npz[k]) for k in sorted(npz.files)], axis=0)
    data = torch.tensor(X, dtype=torch.float32)
    
    print("Training VanillaScore to recreate seed 7777 model...")
    net = train_dsm_vanilla(data, SEED, dev)
    
    print("Performing FFT-based Helmholtz projection...")
    N = 256
    c = torch.linspace(-PI, PI, N+1)[:-1]
    P, S = torch.meshgrid(c, c, indexing="ij")
    q_grid = torch.stack([P.reshape(-1), S.reshape(-1)], dim=-1).to(dev)
    
    with torch.no_grad():
        score = net(q_grid).reshape(N, N, 2)
    
    # We want V such that -\nabla V \approx score => \nabla V \approx -score
    F = -score
    F_phi = F[:,:,0].cpu().numpy()
    F_psi = F[:,:,1].cpu().numpy()
    
    F_phi_hat = np.fft.fft2(F_phi)
    F_psi_hat = np.fft.fft2(F_psi)
    
    # Integer wavenumbers for domain [-pi, pi]
    k = np.fft.fftfreq(N, d=1/N)
    k_phi, k_psi = np.meshgrid(k, k, indexing="ij")
    
    k_sq = k_phi**2 + k_psi**2
    k_sq[0,0] = 1.0 # prevent division by zero
    
    V_hat = (-1j * k_phi * F_phi_hat - 1j * k_psi * F_psi_hat) / k_sq
    V_hat[0,0] = 0.0
    
    V_proj = np.real(np.fft.ifft2(V_hat))
    
    F_irrot_phi = np.real(np.fft.ifft2(1j * k_phi * V_hat))
    F_irrot_psi = np.real(np.fft.ifft2(1j * k_psi * V_hat))
    
    res_phi = F_phi - F_irrot_phi
    res_psi = F_psi - F_irrot_psi
    res_norm = np.sqrt(np.mean(res_phi**2 + res_psi**2))
    
    print("Projection residual (norm of rotational component): %.6e" % res_norm)
    
    # Attractors from vanilla seed 7777
    phi_beta, psi_beta = -73.85, 153.54
    phi_alphaR, psi_alphaR = -76.28, -15.54
    
    def get_V(phi_deg, psi_deg):
        phi_rad = np.deg2rad(phi_deg)
        psi_rad = np.deg2rad(psi_deg)
        i = int(np.round((phi_rad + PI) / (2*PI) * N)) % N
        j = int(np.round((psi_rad + PI) / (2*PI) * N)) % N
        return V_proj[i, j]
        
    v_beta = get_V(phi_beta, psi_beta)
    v_alphaR = get_V(phi_alphaR, psi_alphaR)
    dF = v_alphaR - v_beta
    
    print("V_proj at beta basin (approx %.2f, %.2f): %.4f" % (phi_beta, psi_beta, v_beta))
    print("V_proj at alphaR basin (approx %.2f, %.2f): %.4f" % (phi_alphaR, psi_alphaR, v_alphaR))
    print("Delta F(alphaR - beta) = %.4f kT" % dF)
    
    r3 = (v_beta < v_alphaR - R3_MARGIN)
    print("R3 criterion on V_proj (beta < alphaR - %.2f): %s" % (R3_MARGIN, r3))
    print("WP2_HELMHOLTZ_END")

if __name__ == "__main__":
    main()
