import hashlib, os, sys, platform, math
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import diags
from scipy.sparse.linalg import cg

SCRIPT_VERSION = "v1.1-2026-08-11-claude-EA-seal-A1"
REPO_DIR = r"C:/Project/EquiPhase"
DATA_PATH = os.path.join(REPO_DIR, "data", "ala2", "alanine-dipeptide-3x250ns-backbone-dihedrals.npz")
VANILLA_CKPT = os.path.join(REPO_DIR, "vanilla_deq_seed7777.pt")

ANCHOR_SHA256 = "F5AD30768A7CF3451B3061CB2ECB7F7D1DE8C13044534376D26A6653D4CD5717"
LATENT = 32
PI = math.pi
GRID = 24
DESC_LR = 0.05
DESC_STEPS = 2000
CLUSTER_TOL_DEG = 10.0
MIN_CLUSTER = 3
ANCHORS_DEG = {"beta": (-72.5, +152.5), "alphaR": (-72.5, -17.5)}
R2_TOL_DEG = 25.0
R3_MARGIN = 0.05

E_A_GRID_SIZE = 256
E_A_CG_TOL = 1e-10
E_A_CG_MAXITER = 10000
E_A_THRESH = 0.10

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()

def wrap(x):
    return torch.remainder(x + PI, 2 * PI) - PI

def enc(q):
    phi, psi = q[..., 0:1], q[..., 1:2]
    return torch.cat([torch.sin(phi), torch.cos(phi), torch.sin(psi), torch.cos(psi)], dim=-1)

def per_axis_dist_deg(a_deg, b_deg):
    d = abs(a_deg - b_deg) % 360.0
    return min(d, 360.0 - d)

def macrostate(phi_deg, psi_deg):
    if phi_deg > 0.0: return "alphaL"
    if (psi_deg >= 90.0) or (psi_deg <= -150.0): return "beta"
    if -90.0 <= psi_deg <= 45.0: return "alphaR"
    return "other"

class VanillaForceDEQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(LATENT + 2, 64)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(64, LATENT)

    def force(self, q, x):
        return self.fc2(self.act(self.fc1(torch.cat([q, x], dim=-1))))

class VNet(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(4, h), nn.Tanh(),
                               nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
    def forward(self, q):
        return self.f(enc(q)).squeeze(-1)

def get_S_field(model, dev):
    c = torch.linspace(-PI, PI, E_A_GRID_SIZE + 1)[:-1]
    P, S = torch.meshgrid(c, c, indexing="ij")
    q_grid = torch.stack([P, S], dim=-1).to(dev)
    q_flat = q_grid.reshape(-1, 2)
    
    q_full = torch.zeros((q_flat.shape[0], LATENT), device=dev)
    q_full[:, :2] = q_flat
    x_dummy = torch.zeros((q_flat.shape[0], 2), device=dev)
    
    with torch.no_grad():
        f = model.force(q_full, x_dummy)[:, :2]
        S_field = f.reshape(E_A_GRID_SIZE, E_A_GRID_SIZE, 2)
    return S_field, P, S

def project_P1_spectral(S, dev):
    S_np = S.cpu().numpy()
    S_phi = S_np[..., 0]
    S_psi = S_np[..., 1]
    
    F_phi = np.fft.fft2(S_phi)
    F_psi = np.fft.fft2(S_psi)
    
    # 2pi scaling for [-pi, pi) domain
    # Domain length is 2*pi, so frequencies are scaled by 2*pi / L = 2*pi / 2*pi = 1
    # Actually, standard fftfreq returns cycles per sample spacing (N).
    # Sample spacing is dx = 2*pi / N. So frequencies in radians per unit length are k * (2*pi / L). Since L=2*pi, it's just k.
    # Therefore, integer wave numbers k = np.fft.fftfreq(N) * N.
    k_phi = np.fft.fftfreq(E_A_GRID_SIZE) * E_A_GRID_SIZE
    k_psi = np.fft.fftfreq(E_A_GRID_SIZE) * E_A_GRID_SIZE
    K_phi, K_psi = np.meshgrid(k_phi, k_psi, indexing="ij")
    
    K_sq = K_phi**2 + K_psi**2
    K_sq[0, 0] = 1.0
    
    # S = -grad V  => S = (-dV/dphi, -dV/dpsi)
    # F(S_phi) = -i K_phi F(V) => F(V) = i F(S_phi) / K_phi (for 1D)
    # For 2D: F(V) = i (K_phi * F(S_phi) + K_psi * F(S_psi)) / (K_phi^2 + K_psi^2)
    V_hat = 1j * (K_phi * F_phi + K_psi * F_psi) / K_sq
    V_hat[0, 0] = 0.0
    
    V_P1 = np.real(np.fft.ifft2(V_hat))
    V_P1 = V_P1 - np.mean(V_P1)
    
    return torch.tensor(V_P1, dtype=torch.float32, device=dev)

def project_P2_poisson(S, dev):
    S_np = S.cpu().numpy()
    S_phi = S_np[..., 0]
    S_psi = S_np[..., 1]
    
    dx = 2 * PI / E_A_GRID_SIZE
    dy = 2 * PI / E_A_GRID_SIZE
    
    div_S = (np.roll(S_phi, -1, axis=0) - np.roll(S_phi, 1, axis=0)) / (2 * dx) + \
            (np.roll(S_psi, -1, axis=1) - np.roll(S_psi, 1, axis=1)) / (2 * dy)
    
    # S = -grad V => div S = - Laplacian V => Laplacian V = - div S
    # In discretised form: L V = b, so b = - div_S
    b = -div_S.flatten()
    
    N = E_A_GRID_SIZE
    N2 = N * N
    
    main_diag = -4.0 / (dx * dy) * np.ones(N2)
    off_diag_x = 1.0 / (dx * dx) * np.ones(N2)
    off_diag_y = 1.0 / (dy * dy) * np.ones(N2)
    
    for i in range(1, N):
        off_diag_x[i * N - 1] = 0.0
        
    diagonals = [main_diag, off_diag_x, off_diag_x, off_diag_y, off_diag_y]
    offsets = [0, -1, 1, -N, N]
    
    L = diags(diagonals, offsets, shape=(N2, N2), format="csr")
    
    L_tol = L.tolil()
    for i in range(N):
        L_tol[i, i + N*(N-1)] = 1.0 / (dy * dy)
        L_tol[i + N*(N-1), i] = 1.0 / (dy * dy)
        L_tol[i*N, i*N + N - 1] = 1.0 / (dx * dx)
        L_tol[i*N + N - 1, i*N] = 1.0 / (dx * dx)
    L = L_tol.tocsr()
    
    b = b - np.mean(b)
    
    V_P2_flat, info = cg(L, b, rtol=E_A_CG_TOL, maxiter=E_A_CG_MAXITER)
    if info != 0:
        print(f"Warning: CG did not converge perfectly, info={info}")
        
    V_P2 = V_P2_flat.reshape(N, N)
    V_P2 = V_P2 - np.mean(V_P2)
    
    return torch.tensor(V_P2, dtype=torch.float32, device=dev)

def interpolate_V(V_grid, q):
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

    has_beta = any(a["state"] == "beta" for a in attractors)
    has_alphaR = any(a["state"] == "alphaR" for a in attractors)
    r1 = has_beta and has_alphaR

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

    v_beta, v_alphaR = 999.0, 999.0
    for a in attractors:
        if a["state"] == "beta" and a["V"] < v_beta:
            v_beta = a["V"]
        if a["state"] == "alphaR" and a["V"] < v_alphaR:
            v_alphaR = a["V"]
            
    r3 = False
    if r1 and (v_beta + R3_MARGIN < v_alphaR):
        r3 = True

    dF = None
    if has_beta and has_alphaR:
        dF = v_alphaR - v_beta

    return attractors, r1, r2, r3, dF

def run_v0_test(dev):
    print("\n[V0 SELF-CONSISTENCY UNIT TEST]")
    torch.manual_seed(7777)
    vnet = VNet().to(dev)
    
    c = torch.linspace(-PI, PI, E_A_GRID_SIZE + 1)[:-1]
    P, S = torch.meshgrid(c, c, indexing="ij")
    q_grid = torch.stack([P, S], dim=-1).to(dev)
    q_flat = q_grid.reshape(-1, 2).requires_grad_(True)
    
    V_theta = vnet(q_flat)
    (grad_V,) = torch.autograd.grad(V_theta.sum(), q_flat, create_graph=True)
    S0 = -grad_V.detach().reshape(E_A_GRID_SIZE, E_A_GRID_SIZE, 2)
    V_theta_grid = V_theta.detach().reshape(E_A_GRID_SIZE, E_A_GRID_SIZE)
    V_theta_grid = V_theta_grid - V_theta_grid.mean()
    
    V_P1 = project_P1_spectral(S0, dev)
    V_P2 = project_P2_poisson(S0, dev)
    
    V_P1 = V_P1 - V_P1.mean()
    V_P2 = V_P2 - V_P2.mean()
    
    corr_p1 = np.corrcoef(V_theta_grid.cpu().numpy().flatten(), V_P1.cpu().numpy().flatten())[0, 1]
    mae_p1 = torch.abs(V_theta_grid - V_P1).mean().item()
    
    corr_p2 = np.corrcoef(V_theta_grid.cpu().numpy().flatten(), V_P2.cpu().numpy().flatten())[0, 1]
    mae_p2 = torch.abs(V_theta_grid - V_P2).mean().item()
    
    print(f"  P1 (Spectral): corr={corr_p1:.6f}, MAE={mae_p1:.6f}")
    print(f"  P2 (Poisson):  corr={corr_p2:.6f}, MAE={mae_p2:.6f}")
    
    if corr_p1 < 0.999 or mae_p1 > 0.01 or corr_p2 < 0.999 or mae_p2 > 0.01:
        print("  => V0 FAILED! Aborting execution.")
        sys.exit(1)
    print("  => V0 PASSED! Proceeding to vanilla evaluation.")

def main():
    print("SEAL_EA_HELMHOLTZ_EVAL_BEGIN")
    script_hash = sha256_file(os.path.abspath(__file__))
    print(f"Platform: {platform.platform()} | Python: {sys.version.split()[0]}")
    print(f"Torch: {torch.__version__} | CUDA: {torch.version.cuda} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"[SELF] seal_EA_helmholtz_eval.py SHA-256: {script_hash}")
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    print(f"\n[E-A CONSTANTS]")
    print(f"  E_A_GRID_SIZE = {E_A_GRID_SIZE}")
    print(f"  E_A_CG_TOL = {E_A_CG_TOL}")
    print(f"  E_A_CG_MAXITER = {E_A_CG_MAXITER}")
    print(f"  E_A_THRESH = {E_A_THRESH}")
    
    # Unit test V0
    run_v0_test(dev)
    
    model = VanillaForceDEQ().to(dev)
    model.load_state_dict(torch.load(VANILLA_CKPT, map_location=dev))
    model.eval()
    
    print("\n[E-A PROJECTION]")
    S_field, P, S = get_S_field(model, dev)
    print("  Computing P1 (Spectral Helmholtz)...")
    V_P1 = project_P1_spectral(S_field, dev)
    print("  Computing P2 (CG Poisson)...")
    V_P2 = project_P2_poisson(S_field, dev)
    
    # 3-point check
    print(f"  Field Check points (P1): {V_P1[0,0]:.4f}, {V_P1[128,128]:.4f}, {V_P1[64,192]:.4f}")
    
    print("\n[TO-RUN] P1: Spectral Helmholtz Evaluation")
    attr_p1, r1_p1, r2_p1, r3_p1, dF_p1 = evaluate_V(V_P1, dev)
    print(f"  Attractors found: {len(attr_p1)}")
    for a in attr_p1:
        print(f"    ATTR phi={a['phi']:+8.2f} psi={a['psi']:+8.2f} n={a['n']:3d} state={a['state']:6s} V={a['V']:+.4f}")
    
    df1_str = f"{dF_p1:+.4f}" if dF_p1 is not None else "undefined"
    print(f"  R1={r1_p1} / R2={r2_p1} / R3={r3_p1} / dF={df1_str}")
    
    print("\n[TO-RUN] P2: CG Poisson Evaluation")
    attr_p2, r1_p2, r2_p2, r3_p2, dF_p2 = evaluate_V(V_P2, dev)
    print(f"  Attractors found: {len(attr_p2)}")
    for a in attr_p2:
        print(f"    ATTR phi={a['phi']:+8.2f} psi={a['psi']:+8.2f} n={a['n']:3d} state={a['state']:6s} V={a['V']:+.4f}")
    
    df2_str = f"{dF_p2:+.4f}" if dF_p2 is not None else "undefined"
    print(f"  R1={r1_p2} / R2={r2_p2} / R3={r3_p2} / dF={df2_str}")
    
    print("\n[TO-RUN] Dependency Metrics")
    if dF_p1 is not None and dF_p2 is not None:
        metric_a = abs(dF_p1 - dF_p2)
        print(f"  (a) |dF_P1 - dF_P2| = {metric_a:.6f}")
    else:
        print("  (a) undefined - reason: missing state(s)")
    
    V_P1_norm = V_P1 - V_P1.mean()
    V_P2_norm = V_P2 - V_P2.mean()
    metric_b = torch.abs(V_P1_norm - V_P2_norm).mean().item()
    print(f"  (b) Mean |V_P1_norm - V_P2_norm| = {metric_b:.6f}")
    
    metric_c = (r1_p1 == r1_p2) and (r2_p1 == r2_p2) and (r3_p1 == r3_p2)
    print(f"  (c) P1/P2 R1-R3 Verdict Match = {metric_c}")
    
    print("\nSEAL_EA_HELMHOLTZ_EVAL_END")

if __name__ == "__main__":
    main()
