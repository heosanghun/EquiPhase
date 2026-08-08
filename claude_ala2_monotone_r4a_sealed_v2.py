# -*- coding: utf-8 -*-
# =============================================================================
# claude_ala2_monotone_r4a_sealed_v2.py
# SEALED AUDIT SCRIPT — authored solely by external auditor (Claude, System 2)
# Purpose : R4a re-test under PREREG spec. v1 implementation (auditor error #5)
#           scaled the correction by 0.9*pi (Lip up to ~1.91, NOT a
#           contraction). v2 enforces a true Banach contraction:
#             g(q) = (1-lam)*q + lam*0.9*tanh(SN-MLP(enc(q))),  lam=0.5
#           => Lip(g) <= 0.5 + 0.5*0.9 = 0.95 < 1 (unique fixed point).
#           Iteration in R^2 (no wrap inside dynamics); wrap only for report.
#           Same data anchor, seed 7777, same budget (3000 steps, batch 4096).
# Rules   : Direct invocation only. Submit stdout verbatim. Fail-closed.
# =============================================================================
import hashlib, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn

DATA_PATH = os.path.join("data", "ala2",
                         "alanine-dipeptide-3x250ns-backbone-dihedrals.npz")
ANCHOR_SHA256 = "F5AD30768A7CF3451B3061CB2ECB7F7D1DE8C13044534376D26A6653D4CD5717"
SEED = 7777
SIGMA = 0.15
BATCH = 4096
LR = 1e-3
STEPS = 3000
GRID = 24
ITER_STEPS = 2000
CLUSTER_TOL_DEG = 10.0
MIN_CLUSTER = 3
PI = math.pi

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest().upper()

def wrap(x):
    return torch.remainder(x + PI, 2 * PI) - PI

def enc(q):
    return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)

def per_axis_dist_deg(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)

class MonotoneMapV2(nn.Module):
    def __init__(self, h=64, lam=0.5, scale=0.9):
        super().__init__()
        self.l1 = nn.utils.spectral_norm(nn.Linear(4, h))
        self.l2 = nn.utils.spectral_norm(nn.Linear(h, 2))
        self.lam, self.scale = lam, scale
    def forward(self, q):
        m = self.scale * torch.tanh(self.l2(torch.tanh(self.l1(enc(q)))))
        return (1.0 - self.lam) * q + self.lam * m   # NO wrap inside dynamics

def main():
    t0 = time.strftime("%Y%m%d_%H%M%S")
    print("MONO2_BEGIN")
    print("script=claude_ala2_monotone_r4a_sealed_v2.py run_id=%s" % t0)
    print("torch=%s cuda=%s" % (torch.__version__, torch.cuda.is_available()))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(DATA_PATH):
        print("MONO2_ABORT: data not found"); print("MONO2_END"); sys.exit(1)
    dig = sha256_of(DATA_PATH)
    print("data_sha256=%s" % dig)
    if dig != ANCHOR_SHA256:
        print("MONO2_ABORT: anchor mismatch"); print("MONO2_END"); sys.exit(1)
    print("anchor_check=PASS")
    npz = np.load(DATA_PATH)
    X = np.concatenate([np.asarray(npz[k]) for k in sorted(npz.files)], axis=0)
    data = torch.tensor(X, dtype=torch.float32)
    print("total_samples=%d seed=%d sigma=%.4f batch=%d lr=%.5f steps=%d"
          % (data.shape[0], SEED, SIGMA, BATCH, LR, STEPS))

    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED)
    net = MonotoneMapV2().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    n = data.shape[0]
    for _ in range(STEPS):
        idx = torch.randint(0, n, (BATCH,), generator=g)
        q = data[idx].to(dev)
        eps = torch.randn((BATCH, 2), generator=g).to(dev) * SIGMA
        qt = wrap(q + eps)
        d = net(qt) - q                      # plain R^2 difference (no wrap)
        loss = d.pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    fl = float(loss.item())
    print("final_loss=%.6f" % fl)

    # ---- empirical Lipschitz check (theorem premise verification) ----------
    with torch.no_grad():
        torch.manual_seed(SEED + 1)
        a = (torch.rand(20000, 2, device=dev) * 2 * PI) - PI
        b = a + (torch.randn(20000, 2, device=dev) * 0.05)
        num = (net(a) - net(b)).norm(dim=-1)
        den = (a - b).norm(dim=-1).clamp_min(1e-12)
        lip_emp = float((num / den).max().item())
    print("lipschitz_bound_theory=0.9500 lipschitz_empirical_max=%.6f "
          "contraction=%s" % (lip_emp, lip_emp < 1.0))

    # ---- basin search: iterate g from 24x24 grid ---------------------------
    c = torch.linspace(-PI + PI / GRID, PI - PI / GRID, GRID)
    P, S = torch.meshgrid(c, c, indexing="ij")
    x = torch.stack([P.reshape(-1), S.reshape(-1)], dim=-1).to(dev)
    with torch.no_grad():
        for _ in range(ITER_STEPS):
            x = net(x)
        resid = float((net(x) - x).norm(dim=-1).max().item())
    print("fixed_point_residual_max=%.3e" % resid)
    pts = torch.rad2deg(wrap(x)).cpu().numpy().tolist()

    clusters = []
    for p in pts:
        placed = False
        for cl in clusters:
            if (per_axis_dist_deg(p[0], cl[0][0]) < CLUSTER_TOL_DEG and
                    per_axis_dist_deg(p[1], cl[0][1]) < CLUSTER_TOL_DEG):
                cl[1].append(p); placed = True; break
        if not placed:
            clusters.append((p, [p]))
    attrs = [(cl[0], len(cl[1])) for cl in clusters if len(cl[1]) >= MIN_CLUSTER]
    print("attractors=%d" % len(attrs))
    for (cc, cnt) in attrs:
        print("  ATTR phi=%+8.2f psi=%+8.2f n=%3d" % (cc[0], cc[1], cnt))
    r4a = (len(attrs) == 1)
    print("R4a(monotone N=1)=%s" % r4a)
    print("note: R4b (vanilla) verdict from run 20260808_172910 stands "
          "unchanged (FAIL, preregistered as-is)")
    print("MONO2_END")

if __name__ == "__main__":
    main()
