# -*- coding: utf-8 -*-
# =============================================================================
# claude_ala2_phase2_train_gates_sealed.py
# SEALED AUDIT SCRIPT — authored solely by external auditor (Claude, System 2)
# Spec    : PREREG_ALA2_EQUIPHASE_v1.md (+ v1.1 amendment sealed 2026-08-08:
#           objective=DSM sigma=0.15; batch=4096; adam lr=1e-3; steps=3000;
#           aggregation: EquiPhase R1-R3 PASS iff >=4 of 5 seeds pass;
#           baselines trained on seed 7777 only, same objective/budget)
# Rules   : Direct invocation only. No modification. Submit stdout verbatim.
#           Hash mismatch aborts (fail-closed). New filenames per run.
# Usage   : python claude_ala2_phase2_train_gates_sealed.py
#           (from repository root /home/user/EquiPhase)
# =============================================================================
import hashlib, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn

DATA_PATH = os.path.join("data", "ala2",
                         "alanine-dipeptide-3x250ns-backbone-dihedrals.npz")
ANCHOR_SHA256 = "F5AD30768A7CF3451B3061CB2ECB7F7D1DE8C13044534376D26A6653D4CD5717"
SEEDS = [7777, 1234, 2026, 31415, 65537]
SIGMA = 0.15          # DSM noise (rad)
BATCH = 4096
LR = 1e-3
STEPS = 3000
GRID = 24             # basin-search init grid per axis (24x24=576)
DESC_LR = 0.05
DESC_STEPS = 2000
CLUSTER_TOL_DEG = 10.0
MIN_CLUSTER = 3
ANCHORS_DEG = {"beta": (-72.5, +152.5), "alphaR": (-72.5, -17.5)}
R2_TOL_DEG = 25.0
R3_MARGIN = 0.05      # kT

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
    return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)  # [sinφ,sinψ,cosφ,cosψ]

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

class VanillaScore(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(4, h), nn.ReLU(),
                               nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2))
    def forward(self, q):
        return self.f(enc(q))

class MonotoneMap(nn.Module):
    # g(q) = (1-lam)*q + lam*0.9*pi*tanh(SN-MLP(enc(q))) -> Banach contraction
    def __init__(self, h=64, lam=0.5):
        super().__init__()
        self.l1 = nn.utils.spectral_norm(nn.Linear(4, h))
        self.l2 = nn.utils.spectral_norm(nn.Linear(h, 2))
        self.lam = lam
    def forward(self, q):
        m = 0.9 * PI * torch.tanh(self.l2(torch.tanh(self.l1(enc(q)))))
        return wrap((1.0 - self.lam) * q + self.lam * m)

def grad_v(vnet, q):
    q = q.detach().requires_grad_(True)
    v = vnet(q).sum()
    (g,) = torch.autograd.grad(v, q)
    return g.detach()

def train_dsm_v(data, seed, dev):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = VNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    n = data.shape[0]
    for _ in range(STEPS):
        idx = torch.randint(0, n, (BATCH,), generator=g)
        q = data[idx].to(dev)
        eps = torch.randn((BATCH, 2), generator=g).to(dev) * SIGMA
        qt = wrap(q + eps).requires_grad_(True)
        v = net(qt).sum()
        (gv,) = torch.autograd.grad(v, qt, create_graph=True)
        loss = ((SIGMA ** 2) * gv - eps).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net, float(loss.item())

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
        s = net(qt)                      # model score
        loss = ((SIGMA ** 2) * s + eps).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net, float(loss.item())

def train_monotone(data, seed, dev):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = MonotoneMap().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    n = data.shape[0]
    for _ in range(STEPS):
        idx = torch.randint(0, n, (BATCH,), generator=g)
        q = data[idx].to(dev)
        eps = torch.randn((BATCH, 2), generator=g).to(dev) * SIGMA
        qt = wrap(q + eps)
        d = wrap(net(qt) - q)
        loss = d.pow(2).mean()           # denoising map: g(q_noisy) -> q
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
            # circular mean per axis
            cm = [math.degrees(math.atan2(np.mean(np.sin(np.radians(arr[:, k]))),
                                          np.mean(np.cos(np.radians(arr[:, k])))))
                  for k in (0, 1)]
            out.append({"c": (cm[0], cm[1]), "n": len(cl["m"])})
    out.sort(key=lambda d: -d["n"])
    return out

def basins_from_field(step_fn, dev):
    x = grid_inits(dev)
    for _ in range(DESC_STEPS):
        x = step_fn(x)
    pts = torch.rad2deg(wrap(x)).cpu().numpy().tolist()
    return cluster(pts)

def evaluate_equiphase(vnet, dev):
    def step(x):
        return wrap(x - DESC_LR * grad_v(vnet, x))
    at = basins_from_field(step, dev)
    rows, match = [], {}
    for a in at:
        ms = macrostate(a["c"][0], a["c"][1])
        with torch.no_grad():
            vv = float(vnet(torch.deg2rad(torch.tensor([a["c"]],
                       dtype=torch.float32)).to(dev)).item())
        rows.append((a["c"][0], a["c"][1], a["n"], ms, vv))
    for name, (ap, asx) in ANCHORS_DEG.items():
        best = None
        for (p, s, n, ms, vv) in rows:
            dp, ds = per_axis_dist_deg(p, ap), per_axis_dist_deg(s, asx)
            if dp <= R2_TOL_DEG and ds <= R2_TOL_DEG:
                if best is None or n > best[2]:
                    best = (p, s, n, ms, vv)
        match[name] = best
    states = {ms for (_, _, _, ms, _) in rows}
    r1 = ("beta" in states) and ("alphaR" in states)
    r2 = (match["beta"] is not None) and (match["alphaR"] is not None)
    r3 = (r2 and (match["beta"][4] < match["alphaR"][4] - R3_MARGIN))
    r5 = any(ms == "alphaL" for (_, _, _, ms, _) in rows)
    return rows, r1, r2, r3, r5, match

def main():
    t0 = time.strftime("%Y%m%d_%H%M%S")
    print("GATES_BEGIN")
    print("script=claude_ala2_phase2_train_gates_sealed.py run_id=%s" % t0)
    print("torch=%s cuda=%s" % (torch.__version__, torch.cuda.is_available()))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(DATA_PATH):
        print("GATES_ABORT: data not found"); print("GATES_END"); sys.exit(1)
    dig = sha256_of(DATA_PATH)
    print("data_sha256=%s" % dig)
    if dig != ANCHOR_SHA256:
        print("GATES_ABORT: anchor mismatch"); print("GATES_END"); sys.exit(1)
    print("anchor_check=PASS")
    npz = np.load(DATA_PATH)
    X = np.concatenate([np.asarray(npz[k]) for k in sorted(npz.files)], axis=0)
    data = torch.tensor(X, dtype=torch.float32)
    print("total_samples=%d sigma=%.4f batch=%d lr=%.5f steps=%d" %
          (data.shape[0], SIGMA, BATCH, LR, STEPS))

    os.makedirs("results", exist_ok=True)
    seed_pass = []
    for sd in SEEDS:
        net, fl = train_dsm_v(data, sd, dev)
        torch.save(net.state_dict(),
                   os.path.join("results", "ala2_vnet_seed%d_%s.pt" % (sd, t0)))
        rows, r1, r2, r3, r5, match = evaluate_equiphase(net, dev)
        print("SEED %d | final_loss=%.6f | attractors=%d" % (sd, fl, len(rows)))
        for (p, s, n, ms, vv) in rows:
            print("  ATTR phi=%+8.2f psi=%+8.2f n=%3d state=%-6s V=%+.4f"
                  % (p, s, n, ms, vv))
        if match["beta"] and match["alphaR"]:
            print("  dF(alphaR-beta)=%+.4f kT" %
                  (match["alphaR"][4] - match["beta"][4]))
        print("  R1=%s R2=%s R3=%s R5(exploratory)=%s"
              % (r1, r2, r3, r5))
        seed_pass.append(r1 and r2 and r3)

    npass = sum(seed_pass)
    print("EQUIPHASE aggregate: %d/5 seeds pass R1&R2&R3 -> %s"
          % (npass, "PASS" if npass >= 4 else "FAIL"))

    # ---- baselines (seed 7777) --------------------------------------------
    vs, fl_v = train_dsm_vanilla(data, 7777, dev)
    def vstep(x):
        with torch.no_grad():
            return wrap(x + DESC_LR * vs(x))
    vat = basins_from_field(vstep, dev)
    print("VANILLA seed7777 | final_loss=%.6f | attractors=%d" % (fl_v, len(vat)))
    v_states = set()
    v_r2 = {}
    for a in vat:
        ms = macrostate(a["c"][0], a["c"][1]); v_states.add(ms)
        print("  ATTR phi=%+8.2f psi=%+8.2f n=%3d state=%s"
              % (a["c"][0], a["c"][1], a["n"], ms))
    for name, (ap, asx) in ANCHORS_DEG.items():
        v_r2[name] = any(per_axis_dist_deg(a["c"][0], ap) <= R2_TOL_DEG and
                         per_axis_dist_deg(a["c"][1], asx) <= R2_TOL_DEG
                         for a in vat)
    v_r1 = ("beta" in v_states) and ("alphaR" in v_states)
    v_r12 = v_r1 and v_r2["beta"] and v_r2["alphaR"]
    # exploratory curl diagnostic (not a gate)
    gpts = grid_inits(dev)[:64].requires_grad_(True)
    s_out = vs(gpts)
    j = []
    for k in range(2):
        (gk,) = torch.autograd.grad(s_out[:, k].sum(), gpts, retain_graph=True)
        j.append(gk)
    curl = (j[0][:, 1] - j[1][:, 0]).abs().mean().item()
    print("  vanilla_R1andR2=%s (R4b requires False) | curl_diag=%.4e "
          "(exploratory, not a gate)" % (v_r12, curl))

    mn, fl_m = train_monotone(data, 7777, dev)
    def mstep(x):
        with torch.no_grad():
            return mn(x)
    mat = basins_from_field(mstep, dev)
    print("MONOTONE seed7777 | final_loss=%.6f | attractors=%d" % (fl_m, len(mat)))
    for a in mat:
        print("  ATTR phi=%+8.2f psi=%+8.2f n=%3d" % (a["c"][0], a["c"][1], a["n"]))
    m_r4a = (len(mat) == 1)

    r4 = m_r4a and (not v_r12)
    print("R4a(monotone N=1)=%s R4b(vanilla fails R1&R2)=%s -> R4=%s"
          % (m_r4a, (not v_r12), r4))
    overall = (npass >= 4) and r4
    print("OVERALL: R1-R4 %s (R5 exploratory reported above per seed)"
          % ("PASS" if overall else "FAIL"))
    print("GATES_END")

if __name__ == "__main__":
    main()
