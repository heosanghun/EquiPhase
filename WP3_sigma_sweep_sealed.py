# -*- coding: utf-8 -*-
import hashlib, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn

DATA_PATH = os.path.join("data", "ala2", "alanine-dipeptide-3x250ns-backbone-dihedrals.npz")
SEED = 7777
SIGMAS = [0.05, 0.10, 0.15, 0.25]
ETAS = [0.05, 0.10, 0.20, 0.50, 0.90]
BATCH = 4096
LR = 1e-3
STEPS = 3000
GRID = 24
DESC_DT = 0.05
DESC_STEPS = 2000
CLUSTER_TOL_DEG = 10.0
MIN_CLUSTER = 3
ANCHORS_DEG = {"beta": (-72.5, +152.5), "alphaR": (-72.5, -17.5)}
R2_TOL_DEG = 25.0
R3_MARGIN = 0.05
PI = math.pi

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

class VNet(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(4, h), nn.Tanh(),
                               nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
    def forward(self, q):
        return self.f(enc(q)).squeeze(-1)

def grad_v(vnet, q):
    q = q.detach().requires_grad_(True)
    v = vnet(q).sum()
    (g,) = torch.autograd.grad(v, q)
    return g.detach()

def train_dsm_v(data, seed, sigma, dev):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = VNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    n = data.shape[0]
    for _ in range(STEPS):
        idx = torch.randint(0, n, (BATCH,), generator=g)
        q = data[idx].to(dev)
        eps = torch.randn((BATCH, 2), generator=g).to(dev) * sigma
        qt = wrap(q + eps).requires_grad_(True)
        v = net(qt).sum()
        (gv,) = torch.autograd.grad(v, qt, create_graph=True)
        loss = ((sigma ** 2) * gv - eps).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net, float(loss.item())

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

def evaluate_equiphase_symplectic(vnet, eta, dev):
    q = grid_inits(dev)
    p = torch.zeros_like(q)
    dt = DESC_DT
    
    # 576 points integration
    for _ in range(DESC_STEPS):
        q.requires_grad_(True)
        v = vnet(q).sum()
        (grad_q,) = torch.autograd.grad(v, q)
        p_next = (1.0 - eta) * p - dt * grad_q.detach()
        q_next = wrap(q.detach() + dt * p_next)
        q, p = q_next, p_next
        
    # Check divergence
    nan_mask = torch.isnan(q).any(dim=-1)
    inf_mask = torch.isinf(q).any(dim=-1)
    p_div_mask = (p.abs() > 1e4).any(dim=-1)
    diverged = nan_mask | inf_mask | p_div_mask
    
    q_conv = q[~diverged]
    converged_count = int((~diverged).sum().item())
    conv_rate = converged_count / (GRID * GRID)
    
    pts = torch.rad2deg(wrap(q_conv)).cpu().numpy().tolist()
    at = cluster(pts)
    
    raw_rows = []
    for a in at:
        with torch.no_grad():
            vv = float(vnet(torch.deg2rad(torch.tensor([a["c"]], dtype=torch.float32)).to(dev)).item())
        raw_rows.append((a["c"][0], a["c"][1], a["n"], vv))
        
    if len(raw_rows) > 0:
        v_min = min(r[3] for r in raw_rows)
    else:
        v_min = 0.0
        
    rows, match = [], {}
    for (phi_deg, psi_deg, n, vv) in raw_rows:
        ms = macrostate(phi_deg, psi_deg)
        is_artifact = (vv - v_min > 10.0)
        if is_artifact:
            ms = "artifact"
        rows.append((phi_deg, psi_deg, n, ms, vv))
        
    for name, (ap, asx) in ANCHORS_DEG.items():
        best = None
        for (pos_phi, pos_psi, n, ms, vv) in rows:
            if ms == "artifact":
                continue
            dp, ds = per_axis_dist_deg(pos_phi, ap), per_axis_dist_deg(pos_psi, asx)
            if dp <= R2_TOL_DEG and ds <= R2_TOL_DEG:
                if best is None or n > best[2]:
                    best = (pos_phi, pos_psi, n, ms, vv)
        match[name] = best
        
    states = {ms for (_, _, _, ms, _) in rows if ms != "artifact"}
    r1 = ("beta" in states) and ("alphaR" in states)
    r2 = (match["beta"] is not None) and (match["alphaR"] is not None)
    r3 = (r2 and (match["beta"][4] < match["alphaR"][4] - R3_MARGIN))
    r5 = any(ms == "alphaL" for (_, _, _, ms, _) in rows if ms != "artifact")
    artifacts = any(ms == "artifact" or ms == "other" for (_, _, _, ms, _) in rows)
    
    return rows, r1, r2, r3, r5, artifacts, conv_rate, match

def main():
    print("WP3_SIGMA_SWEEP_BEGIN")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    npz = np.load(DATA_PATH)
    X = np.concatenate([np.asarray(npz[k]) for k in sorted(npz.files)], axis=0)
    data = torch.tensor(X, dtype=torch.float32)
    
    for sig in SIGMAS:
        print("Training VNet for SIGMA=%.4f..." % sig)
        net, fl = train_dsm_v(data, SEED, sig, dev)
        for eta in ETAS:
            rows, r1, r2, r3, r5, artifacts, conv_rate, match = evaluate_equiphase_symplectic(net, eta, dev)
            print("SEED %d | SIGMA=%.4f | ETA=%.2f | final_loss=%.6f | ConvRate=%.1f%% | attractors=%d" % (
                SEED, sig, eta, fl, conv_rate*100, len(rows)))
            for (p, s, n, ms, vv) in rows:
                print("  ATTR phi=%+8.2f psi=%+8.2f n=%3d state=%-8s V=%+.4f" % (p, s, n, ms, vv))
            if match["beta"] and match["alphaR"]:
                dF = match["alphaR"][4] - match["beta"][4]
                print("  dF(alphaR-beta)=%+.4f kT" % dF)
            print("  R1=%s R2=%s R3=%s alphaL_detect=%s artifact_presence=%s" % (r1, r2, r3, r5, artifacts))
            
    print("WP3_SIGMA_SWEEP_END")

if __name__ == "__main__":
    main()
