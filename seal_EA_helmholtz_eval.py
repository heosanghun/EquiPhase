import hashlib
import os
import sys
import platform
import math
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import diags
from scipy.sparse.linalg import cg

SCRIPT_VERSION = "v1.0-2026-08-11-claude-EA-seal"
REPO_DIR = r"C:/Project/EquiPhase"
DATA_PATH = os.path.join(REPO_DIR, "data", "ala2", "alanine-dipeptide-3x250ns-backbone-dihedrals.npz")
VANILLA_CKPT = os.path.join(REPO_DIR, "vanilla_deq_seed7777.pt")

ANCHOR_SHA256 = "F5AD30768A7CF3451B3061CB2ECB7F7D1DE8C13044534376D26A6653D4CD5717"
LATENT = 32
PI = math.pi
GRID = 24             # basin-search init grid per axis (24x24=576)
DESC_LR = 0.05
DESC_STEPS = 2000
ANCHORS_DEG = {"beta": (-72.5, +152.5), "alphaR": (-72.5, -17.5)}
R2_TOL_DEG = 25.0
R3_MARGIN = 0.05      # kT
CLUSTER_TOL_DEG = 10.0
MIN_CLUSTER = 3

E_A_GRID_SIZE = 256
E_A_CG_TOL = 1e-10
E_A_CG_MAXITER = 10000
E_A_THRESH = 0.10

SEP = "=" * 88

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()

def wrap(x):
    return torch.remainder(x + PI, 2 * PI) - PI

def enc(q):
    return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)

def per_axis_dist_deg(a_deg, b_deg):
    d = abs(a_deg - b_deg) % 360.0
    return min(d, 360.0 - d)

def macrostate(phi_deg, psi_deg):
    if phi_deg > 0.0:
        return "alphaL"
    if (psi_deg >= 90.0) or (psi_deg <= -150.0):
        return "beta"
    if -90.0 <= psi_deg <= 45.0:
        return "alphaR"
    return "other"

class VanillaForceDEQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(LATENT + 2, 64)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(64, LATENT)

    def force(self, q, x):
        return self.fc2(self.act(self.fc1(torch.cat([q, x], dim=-1))))

def get_S_field(model, dev):
    c = torch.linspace(-PI, PI, E_A_GRID_SIZE + 1)[:-1]
    P, S = torch.meshgrid(c, c, indexing="ij")
    q_grid = torch.stack([P, S], dim=-1).to(dev)
    q_flat = q_grid.reshape(-1, 2)
    
    # Pad to LATENT
    q_full = torch.zeros((q_flat.shape[0], LATENT), device=dev)
    q_full[:, :2] = q_flat
    x_dummy = torch.zeros((q_flat.shape[0], 2), device=dev)
    
    with torch.no_grad():
        f = model.force(q_full, x_dummy)[:, :2]
        # In the context of DEQ Score Matching, force is the score. S = f
        S_field = f.reshape(E_A_GRID_SIZE, E_A_GRID_SIZE, 2)
    return S_field, P, S

def project_P1_spectral(S, dev):
    S_np = S.cpu().numpy()
    S_phi = S_np[..., 0]
    S_psi = S_np[..., 1]
    
    F_phi = np.fft.fft2(S_phi)
    F_psi = np.fft.fft2(S_psi)
    
    k_phi = np.fft.fftfreq(E_A_GRID_SIZE) * E_A_GRID_SIZE
    k_psi = np.fft.fftfreq(E_A_GRID_SIZE) * E_A_GRID_SIZE
    K_phi, K_psi = np.meshgrid(k_phi, k_psi, indexing="ij")
    
    K_sq = K_phi**2 + K_psi**2
    K_sq[0, 0] = 1.0 # Avoid div by 0
    
    # S_cf(k) = k (k dot S(k)) / |k|^2
    # V(k) = i k dot S(k) / |k|^2  (since S = -grad V)
    
    V_hat = 1j * (K_phi * F_phi + K_psi * F_psi) / K_sq
    V_hat[0, 0] = 0.0
    
    V_P1 = np.real(np.fft.ifft2(V_hat))
    # Normalize mean to 0
    V_P1 = V_P1 - np.mean(V_P1)
    
    return torch.tensor(V_P1, dtype=torch.float32, device=dev)

def project_P2_poisson(S, dev):
    S_np = S.cpu().numpy()
    S_phi = S_np[..., 0]
    S_psi = S_np[..., 1]
    
    dx = 2 * PI / E_A_GRID_SIZE
    dy = 2 * PI / E_A_GRID_SIZE
    
    # div S = dS_phi/dphi + dS_psi/dpsi
    div_S = (np.roll(S_phi, -1, axis=0) - np.roll(S_phi, 1, axis=0)) / (2 * dx) + \
            (np.roll(S_psi, -1, axis=1) - np.roll(S_psi, 1, axis=1)) / (2 * dy)
    b = -div_S.flatten()
    
    N = E_A_GRID_SIZE
    N2 = N * N
    
    # 5-point Laplacian with periodic BC
    main_diag = -4.0 / (dx * dy) * np.ones(N2)
    off_diag_x = 1.0 / (dx * dx) * np.ones(N2)
    off_diag_y = 1.0 / (dy * dy) * np.ones(N2)
    
    # Zero out boundaries for off_diag_x
    for i in range(1, N):
        off_diag_x[i * N - 1] = 0.0
        
    diagonals = [main_diag, off_diag_x[1:], off_diag_x[1:], off_diag_y[N:], off_diag_y[N:]]
    offsets = [0, -1, 1, -N, N]
    
    L = diags(diagonals, offsets, shape=(N2, N2), format="csr")
    
    # Add periodic BC
    L_tol = L.tolil()
    for i in range(N):
        L_tol[i, i + N*(N-1)] = 1.0 / (dy * dy)
        L_tol[i + N*(N-1), i] = 1.0 / (dy * dy)
        L_tol[i*N, i*N + N - 1] = 1.0 / (dx * dx)
        L_tol[i*N + N - 1, i*N] = 1.0 / (dx * dx)
    L = L_tol.tocsr()
    
    # To fix gauge, set V(0)=0 by modifying equation 0
    L[0, :] = 0
    L[0, 0] = 1.0
    b[0] = 0.0
    
    V_P2_flat, info = cg(L, b, rtol=E_A_CG_TOL, maxiter=E_A_CG_MAXITER)
    if info != 0:
        print(f"Warning: CG did not converge perfectly, info={info}")
        
    V_P2 = V_P2_flat.reshape(N, N)
    
    # Gauge fix: beta anchor = 0
    idx_phi = int((-72.5 + 180.0) / 360.0 * N)
    idx_psi = int((152.5 + 180.0) / 360.0 * N)
    idx_phi = min(max(idx_phi, 0), N-1)
    idx_psi = min(max(idx_psi, 0), N-1)
    
    V_P2 = V_P2 - V_P2[idx_phi, idx_psi]
    
    return torch.tensor(V_P2, dtype=torch.float32, device=dev)

def interpolate_V(V_grid, q):
    # bilinear interpolation
    q_wrap = wrap(q)
    u = (q_wrap[:, 0] + PI) / (2 * PI) * E_A_GRID_SIZE
    v = (q_wrap[:, 1] + PI) / (2 * PI) * E_A_GRID_SIZE
    
    u = u % E_A_GRID_SIZE
    v = v % E_A_GRID_SIZE
    
    u0 = torch.floor(u).long()
    v0 = torch.floor(v).long()
    u1 = (u0 + 1) % E_A_GRID_SIZE
    v1 = (v0 + 1) % E_A_GRID_SIZE
    
    du = u - u0.float()
    dv = v - v0.float()
    
    V00 = V_grid[u0, v0]
    V10 = V_grid[u1, v0]
    V01 = V_grid[u0, v1]
    V11 = V_grid[u1, v1]
    
    V_interp = V00 * (1 - du) * (1 - dv) + \
               V10 * du * (1 - dv) + \
               V01 * (1 - du) * dv + \
               V11 * du * dv
               
    return V_interp

def grad_V_interp(V_grid, q):
    q = q.detach().requires_grad_(True)
    v = interpolate_V(V_grid, q).sum()
    (g,) = torch.autograd.grad(v, q)
    return g.detach()

def grid_inits(dev):
    c = torch.linspace(-PI + PI / GRID, PI - PI / GRID, GRID)
    P, S = torch.meshgrid(c, c, indexing="ij")
    return torch.stack([P.reshape(-1), S.reshape(-1)], dim=-1).to(dev)

def evaluate_V(V_grid, dev):
    q0 = grid_inits(dev)
    q = q0.clone()
    for _ in range(DESC_STEPS):
        g = grad_V_interp(V_grid, q)
        q = wrap(q - DESC_LR * g)

    q_deg = q * 180.0 / PI
    attractors = []
    
    for i in range(q.shape[0]):
        pd, sd = float(q_deg[i, 0]), float(q_deg[i, 1])
        found = False
        for a in attractors:
            if per_axis_dist_deg(pd, a["phi"]) < CLUSTER_TOL_DEG and \
               per_axis_dist_deg(sd, a["psi"]) < CLUSTER_TOL_DEG:
                a["n"] += 1
                found = True
                break
        if not found:
            attractors.append({"phi": pd, "psi": sd, "n": 1,
                               "state": macrostate(pd, sd),
                               "q_rad": q[i:i+1]})

    attractors = [a for a in attractors if a["n"] >= MIN_CLUSTER]
    for a in attractors:
        a["V"] = interpolate_V(V_grid, a["q_rad"]).item()

    # R1: beta and alphaR exist
    has_beta = any(a["state"] == "beta" for a in attractors)
    has_alphaR = any(a["state"] == "alphaR" for a in attractors)
    r1 = has_beta and has_alphaR

    # R2: close to anchors
    r2_beta = False
    for a in attractors:
        if a["state"] == "beta":
            d_phi = per_axis_dist_deg(a["phi"], ANCHORS_DEG["beta"][0])
            d_psi = per_axis_dist_deg(a["psi"], ANCHORS_DEG["beta"][1])
            if d_phi <= R2_TOL_DEG and d_psi <= R2_TOL_DEG:
                r2_beta = True
                break
    r2_alphaR = False
    for a in attractors:
        if a["state"] == "alphaR":
            d_phi = per_axis_dist_deg(a["phi"], ANCHORS_DEG["alphaR"][0])
            d_psi = per_axis_dist_deg(a["psi"], ANCHORS_DEG["alphaR"][1])
            if d_phi <= R2_TOL_DEG and d_psi <= R2_TOL_DEG:
                r2_alphaR = True
                break
    r2 = r2_beta and r2_alphaR

    # R3: depth ordering
    v_beta, v_alphaR = 999.0, 999.0
    for a in attractors:
        if a["state"] == "beta" and a["V"] < v_beta:
            v_beta = a["V"]
        if a["state"] == "alphaR" and a["V"] < v_alphaR:
            v_alphaR = a["V"]
    r3 = False
    dF = v_alphaR - v_beta
    if r1 and (v_beta + R3_MARGIN < v_alphaR):
        r3 = True

    return attractors, r1, r2, r3, dF

def main():
    print("SEAL_EA_HELMHOLTZ_EVAL_BEGIN")
    script_hash = sha256_file(os.path.abspath(__file__))
    print(f"Platform: {platform.platform()} | Python: {sys.version.split()[0]}")
    print(f"Torch: {torch.__version__} | CUDA: {torch.version.cuda} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"[SELF] seal_EA_helmholtz_eval.py SHA-256: {script_hash}")
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Input Integrity
    if not os.path.exists(DATA_PATH):
        print(f"ABORT: Missing data at {DATA_PATH}")
        sys.exit(1)
    if not os.path.exists(VANILLA_CKPT):
        print(f"ABORT: Missing checkpoint at {VANILLA_CKPT}")
        sys.exit(1)
        
    data_hash = sha256_file(DATA_PATH)
    if data_hash != ANCHOR_SHA256:
        print(f"ABORT: Data hash mismatch! {data_hash} != {ANCHOR_SHA256}")
        sys.exit(1)
    print(f"Data SHA-256 Verified: {data_hash}")
    
    ckpt_hash = sha256_file(VANILLA_CKPT)
    print(f"Vanilla Checkpoint SHA-256: {ckpt_hash}")
    
    # 2. Echo Constants
    print(f"\n[E-A CONSTANTS]")
    print(f"  E_A_GRID_SIZE = {E_A_GRID_SIZE}")
    print(f"  E_A_CG_TOL = {E_A_CG_TOL}")
    print(f"  E_A_CG_MAXITER = {E_A_CG_MAXITER}")
    print(f"  E_A_THRESH = {E_A_THRESH}")
    
    # 3. Load Model
    model = VanillaForceDEQ().to(dev)
    model.load_state_dict(torch.load(VANILLA_CKPT, map_location=dev))
    model.eval()
    
    # 4. Extract Field and Project
    print("\n[E-A PROJECTION]")
    S_field, P, S = get_S_field(model, dev)
    print("  Computing P1 (Spectral Helmholtz)...")
    V_P1 = project_P1_spectral(S_field, dev)
    print("  Computing P2 (CG Poisson)...")
    V_P2 = project_P2_poisson(S_field, dev)
    
    # 5. Evaluate P1
    print("\n[TO-RUN] P1: Spectral Helmholtz Evaluation")
    attr_p1, r1_p1, r2_p1, r3_p1, dF_p1 = evaluate_V(V_P1, dev)
    print(f"  Attractors found: {len(attr_p1)}")
    for a in attr_p1:
        print(f"    ATTR phi={a['phi']:+8.2f} psi={a['psi']:+8.2f} n={a['n']:3d} state={a['state']:6s} V={a['V']:+.4f}")
    print(f"  R1={r1_p1} / R2={r2_p1} / R3={r3_p1} / dF={dF_p1:+.4f}")
    
    # 6. Evaluate P2
    print("\n[TO-RUN] P2: CG Poisson Evaluation")
    attr_p2, r1_p2, r2_p2, r3_p2, dF_p2 = evaluate_V(V_P2, dev)
    print(f"  Attractors found: {len(attr_p2)}")
    for a in attr_p2:
        print(f"    ATTR phi={a['phi']:+8.2f} psi={a['psi']:+8.2f} n={a['n']:3d} state={a['state']:6s} V={a['V']:+.4f}")
    print(f"  R1={r1_p2} / R2={r2_p2} / R3={r3_p2} / dF={dF_p2:+.4f}")
    
    # 7. Dependency Metrics
    print("\n[TO-RUN] Dependency Metrics")
    metric_a = abs(dF_p1 - dF_p2)
    print(f"  (a) |dF_P1 - dF_P2| = {metric_a:.6f}")
    
    # Normalize means to 0 for metric (b)
    V_P1_norm = V_P1 - V_P1.mean()
    V_P2_norm = V_P2 - V_P2.mean()
    metric_b = torch.abs(V_P1_norm - V_P2_norm).mean().item()
    print(f"  (b) Mean |V_P1_norm - V_P2_norm| = {metric_b:.6f}")
    
    metric_c = (r1_p1 == r1_p2) and (r2_p1 == r2_p2) and (r3_p1 == r3_p2)
    print(f"  (c) P1/P2 R1-R3 Verdict Match = {metric_c}")
    
    # 8. Verdict Interpretation
    # print("\n[TO-RUN] Verdict Interpretation")
    # supports_claim = (not metric_c) or (metric_a > E_A_THRESH)
    # if supports_claim:
    #     print(f"  Result Interpretation A: Projection dependency verified. The claim is supported.")
    # else:
    #     print(f"  Result Interpretation B: Projection dependency weak. The claim must be softened.")
        
    print("\nSEAL_EA_HELMHOLTZ_EVAL_END")

if __name__ == "__main__":
    main()
