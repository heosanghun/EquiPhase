# -*- coding: utf-8 -*-
import hashlib, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

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
    
    V_spectral = np.real(np.fft.ifft2(V_hat))
    V_spectral -= np.mean(V_spectral)
    
    F_irrot_phi = np.real(np.fft.ifft2(1j * k_phi * V_hat))
    F_irrot_psi = np.real(np.fft.ifft2(1j * k_psi * V_hat))
    
    res_phi = F_phi - F_irrot_phi
    res_psi = F_psi - F_irrot_psi
    res_norm = np.sqrt(np.mean(res_phi**2 + res_psi**2))
    
    print("Projection residual (norm of rotational component): %.6e" % res_norm)
    
    # 2. Finite-Difference Solver
    print("Performing Finite-Difference Poisson solver...")
    h = 2.0 * PI / N
    g = np.zeros((N, N))
    for i in range(N):
        ip1 = (i + 1) % N
        im1 = (i - 1) % N
        for j in range(N):
            jp1 = (j + 1) % N
            jm1 = (j - 1) % N
            g[i, j] = 0.5 * h * (F_phi[ip1, j] - F_phi[im1, j] + F_psi[i, jp1] - F_psi[i, jm1])
            
    rows = []
    cols = []
    data = []
    for i in range(N):
        for j in range(N):
            k_idx = i * N + j
            if i == 0 and j == 0:
                rows.append(k_idx)
                cols.append(k_idx)
                data.append(1.0)
                continue
            ip1_k = ((i + 1) % N) * N + j
            im1_k = ((i - 1) % N) * N + j
            jp1_k = i * N + ((j + 1) % N)
            jm1_k = i * N + ((j - 1) % N)
            
            rows.extend([k_idx, k_idx, k_idx, k_idx, k_idx])
            cols.extend([k_idx, ip1_k, im1_k, jp1_k, jm1_k])
            data.extend([-4.0, 1.0, 1.0, 1.0, 1.0])
            
    g_flat = g.flatten()
    g_flat[0] = 0.0
    
    A = sp.coo_matrix((data, (rows, cols)), shape=(N*N, N*N)).tocsr()
    V_flat = spsolve(A, g_flat)
    V_fd = V_flat.reshape(N, N)
    V_fd -= np.mean(V_fd)
    
    # Attractors from vanilla seed 7777
    phi_beta, psi_beta = -73.85, 153.54
    phi_alphaR, psi_alphaR = -76.28, -15.54
    
    def get_V(V_arr, phi_deg, psi_deg):
        phi_rad = np.deg2rad(phi_deg)
        psi_rad = np.deg2rad(psi_deg)
        i = int(np.round((phi_rad + PI) / (2*PI) * N)) % N
        j = int(np.round((psi_rad + PI) / (2*PI) * N)) % N
        return V_arr[i, j]
        
    v_beta_spec = get_V(V_spectral, phi_beta, psi_beta)
    v_alphaR_spec = get_V(V_spectral, phi_alphaR, psi_alphaR)
    dF_spec = v_alphaR_spec - v_beta_spec
    
    v_beta_fd = get_V(V_fd, phi_beta, psi_beta)
    v_alphaR_fd = get_V(V_fd, phi_alphaR, psi_alphaR)
    dF_fd = v_alphaR_fd - v_beta_fd
    
    print("V_proj (Spectral) at beta basin: %.4f" % v_beta_spec)
    print("V_proj (Spectral) at alphaR basin: %.4f" % v_alphaR_spec)
    print("Delta F (Spectral) = %.4f kT" % dF_spec)
    
    print("V_proj (FD) at beta basin: %.4f" % v_beta_fd)
    print("V_proj (FD) at alphaR basin: %.4f" % v_alphaR_fd)
    print("Delta F (FD) = %.4f kT" % dF_fd)
    
    discrepancy = abs(dF_spec - dF_fd)
    print("Discrepancy in Delta F: %.4f kT" % discrepancy)
    
    # 3. Path Integration Solver to demonstrate non-conservativeness
    print("Performing Path-based Line Integrals...")
    phi_b, psi_b = np.deg2rad(phi_beta), np.deg2rad(psi_beta)
    phi_aR, psi_aR = np.deg2rad(phi_alphaR), np.deg2rad(psi_alphaR)
    
    def trapz(y, x):
        dx = x[1] - x[0]
        return np.sum(0.5 * (y[:-1] + y[1:])) * dx
        
    steps = 1000
    t1 = np.linspace(phi_b, phi_aR, steps)
    q1 = torch.tensor([[t, psi_b] for t in t1], dtype=torch.float32).to(dev)
    with torch.no_grad():
        F1 = -net(q1).cpu().numpy()
    int_A1 = trapz(F1[:, 0], t1)
    
    dpsi = float(wrap(torch.tensor(psi_aR - psi_b)).item())
    t2 = np.linspace(psi_b, psi_b + dpsi, steps)
    t2_wrapped = np.array([float(wrap(torch.tensor(t)).item()) for t in t2])
    q2 = torch.tensor([[phi_aR, t] for t in t2_wrapped], dtype=torch.float32).to(dev)
    with torch.no_grad():
        F2 = -net(q2).cpu().numpy()
    int_A2 = trapz(F2[:, 1], t2)
    dF_pathA = int_A1 + int_A2
    
    t3 = np.linspace(psi_b, psi_b + dpsi, steps)
    t3_wrapped = np.array([float(wrap(torch.tensor(t)).item()) for t in t3])
    q3 = torch.tensor([[phi_b, t] for t in t3_wrapped], dtype=torch.float32).to(dev)
    with torch.no_grad():
        F3 = -net(q3).cpu().numpy()
    int_B1 = trapz(F3[:, 1], t3)
    
    t4 = np.linspace(phi_b, phi_aR, steps)
    q4 = torch.tensor([[t, psi_aR] for t in t4], dtype=torch.float32).to(dev)
    with torch.no_grad():
        F4 = -net(q4).cpu().numpy()
    int_B2 = trapz(F4[:, 0], t4)
    dF_pathB = int_B1 + int_B2
    
    print("Delta F (Path A) = %.4f kT" % dF_pathA)
    print("Delta F (Path B) = %.4f kT" % dF_pathB)
    print("Discrepancy (Path A vs Path B) = %.4f kT" % abs(dF_pathA - dF_pathB))
    print("Discrepancy (Poisson vs Path A) = %.4f kT" % abs(dF_spec - dF_pathA))
    print("Discrepancy (Poisson vs Path B) = %.4f kT" % abs(dF_spec - dF_pathB))
    
    r3 = (v_beta_spec < v_alphaR_spec - R3_MARGIN)
    print("R3 criterion on V_proj (beta < alphaR - %.2f): %s" % (R3_MARGIN, r3))
    print("WP2_HELMHOLTZ_END")

if __name__ == "__main__":
    main()
