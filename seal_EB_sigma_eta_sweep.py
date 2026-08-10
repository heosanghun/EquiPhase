import hashlib, math, os, sys, time, platform
import numpy as np
import torch
import torch.nn as nn

def get_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

DATA_PATH = os.path.join("data", "ala2", "alanine-dipeptide-3x250ns-backbone-dihedrals.npz")
SEEDS = [7777, 8888, 9999, 1111, 2222]
SIGMAS = [0.05, 0.10, 0.15, 0.25]
BATCH = 4096
LR = 1e-3
STEPS = 3000
GRID = 24
ETAS = [0.05, 0.10, 0.20, 0.50, 0.90]
DESC_DT = 0.05
DESC_STEPS = 2000
CLUSTER_TOL_DEG = 10.0
MIN_CLUSTER = 3
ANCHORS_DEG = {"beta": (-72.5, +152.5), "alphaR": (-72.5, -17.5)}
R2_TOL_DEG = 25.0
R3_MARGIN = 0.5  # PI-approved depth-margin
PI = math.pi

def wrap(x):
    return torch.remainder(x + PI, 2 * PI) - PI

def enc(q):
    return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)

def per_axis_dist_deg(a_deg, b_deg):
    d = abs(a_deg - b_deg) % 360.0
    return min(d, 360.0 - d)

def macrostate(phi_deg, psi_deg):
    if phi_deg > 0.0: return "alphaL"
    if (psi_deg >= 90.0) or (psi_deg <= -150.0): return "beta"
    if -90.0 <= psi_deg <= 45.0: return "alphaR"
    return "other"

class VNet(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(4, h), nn.Tanh(),
                               nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
    def forward(self, q):
        return self.f(enc(q)).squeeze(-1)

def train_dsm_v(data, seed, sigma, dev):
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    net = VNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    n = data.shape[0]
    for _ in range(STEPS):
        idx = torch.randint(0, n, (BATCH,), generator=g)
        q = data[idx].to(dev)
        # We generate eps on cpu first if g is cpu generator, but let's just use torch.randn and move it.
        # Actually it's fine to just generate eps on dev if we don't care about strict determinism across devices.
        # To maintain strict determinism, better generate on CPU then move.
        eps = torch.randn((BATCH, 2), generator=g, device="cpu").to(dev) * sigma
        qt = wrap(q + eps).requires_grad_(True)
        v = net(qt).sum()
        (gv,) = torch.autograd.grad(v, qt, create_graph=True)
        loss = ((sigma ** 2) * gv - eps).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net

def grid_inits(dev):
    c = torch.linspace(-PI + PI / GRID, PI - PI / GRID, GRID)
    P, S = torch.meshgrid(c, c, indexing="ij")
    return torch.stack([P.reshape(-1), S.reshape(-1)], dim=-1).to(dev)

def cluster(points_deg):
    clusters = []
    for p in points_deg:
        placed = False
        for cl in clusters:
            if (per_axis_dist_deg(p[0], cl["c"][0]) < CLUSTER_TOL_DEG and
                    per_axis_dist_deg(p[1], cl["c"][1]) < CLUSTER_TOL_DEG):
                cl["m"].append(p); placed = True; break
        if not placed:
            clusters.append({"c": p, "m": [p]})
    out = []
    for cl in clusters:
        if len(cl["m"]) >= MIN_CLUSTER:
            arr = np.array(cl["m"])
            cm = [math.degrees(math.atan2(np.mean(np.sin(np.radians(arr[:, k]))),
                                          np.mean(np.cos(np.radians(arr[:, k])))))
                  for k in (0, 1)]
            out.append({"c": (cm[0], cm[1]), "n": len(cl["m"])})
    out.sort(key=lambda d: -d["n"])
    return out

def evaluate_symplectic_eta(vnet, eta, dev):
    q = grid_inits(dev)
    p = torch.zeros_like(q)
    dt = DESC_DT
    for _ in range(DESC_STEPS):
        q.requires_grad_(True)
        v = vnet(q).sum()
        (grad_q,) = torch.autograd.grad(v, q)
        p_next = (1.0 - eta) * p - dt * grad_q.detach()
        q_next = wrap(q.detach() + dt * p_next)
        q, p = q_next, p_next
        if torch.isnan(q).any() or torch.isinf(q).any():
            return None, False, False, False, 0.0
            
    pts = torch.rad2deg(wrap(q)).cpu().numpy().tolist()
    at = cluster(pts)
    rows, match = [], {}
    for a in at:
        ms = macrostate(a["c"][0], a["c"][1])
        with torch.no_grad():
            vv = float(vnet(torch.deg2rad(torch.tensor([a["c"]], dtype=torch.float32)).to(dev)).item())
        rows.append((a["c"][0], a["c"][1], a["n"], ms, vv))
        
    for name, (ap, asx) in ANCHORS_DEG.items():
        best = None
        for (pos_phi, pos_psi, n, ms, vv) in rows:
            dp, ds = per_axis_dist_deg(pos_phi, ap), per_axis_dist_deg(pos_psi, asx)
            if dp <= R2_TOL_DEG and ds <= R2_TOL_DEG:
                if best is None or n > best[2]:
                    best = (pos_phi, pos_psi, n, ms, vv)
        match[name] = best
        
    states = {ms for (_, _, _, ms, _) in rows}
    r1 = ("beta" in states) and ("alphaR" in states)
    r2 = (match["beta"] is not None) and (match["alphaR"] is not None)
    r3 = (r2 and (match["beta"][4] < match["alphaR"][4] - R3_MARGIN))
    
    converged_pts = sum(a["n"] for a in at)
    convergence_rate = converged_pts / (GRID * GRID)
    return rows, r1, r2, r3, convergence_rate

def main():
    print("SEAL_EB_SIGMA_ETA_SWEEP_BEGIN")
    print(f"Platform: {platform.platform()} | Python: {sys.version.split()[0]}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    npz = np.load(DATA_PATH)
    X = np.concatenate([np.asarray(npz[k]) for k in sorted(npz.files)], axis=0)
    data = torch.tensor(X, dtype=torch.float32)
    
    diverging_cases = []
    
    for seed in SEEDS:
        for sigma in SIGMAS:
            print(f"\n--- SEED {seed} | SIGMA {sigma:.2f} ---")
            net = train_dsm_v(data, seed, sigma, dev)
            for eta in ETAS:
                rows, r1, r2, r3, conv_rate = evaluate_symplectic_eta(net, eta, dev)
                if rows is None:
                    print(f"  ETA={eta:.2f} | DIVERGED (NaN/Inf)")
                    diverging_cases.append((seed, sigma, eta))
                else:
                    print(f"  ETA={eta:.2f} | ConvRate={conv_rate*100:.1f}% | attractors={len(rows)} | R1={r1} R2={r2} R3={r3}")
                    
    print(f"\n--- DIVERGING CASES (Total: {len(diverging_cases)}) ---")
    for (s, sig, e) in diverging_cases:
        print(f"Seed={s}, Sigma={sig:.2f}, Eta={e:.2f}")

    script_hash = get_hash(__file__)
    print(f"\n[SELF] seal_EB_sigma_eta_sweep.py SHA-256: {script_hash}")
    print("SEAL_EB_SIGMA_ETA_SWEEP_END")

if __name__ == "__main__":
    main()
