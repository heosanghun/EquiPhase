# =============================================================================
# claude_paper2_baselines_sealed.py
# Author  : Claude (external auditor, System 2)  --  SEALED SCRIPT, DO NOT MODIFY
# Version : v1.0 (2026-08-08)
# Purpose : Step 12 baseline comparison for Paper 2 (EquiPhase DEQ).
#           Trains and audits two control models under identical protocol:
#             Baseline 1  Vanilla DEQ   (unconstrained force field MLP 34->64->32)
#             Baseline 2  Monotone DEQ  (contraction map, ||W||_2 <= 0.9)
#
# INTEGRITY MODEL (same as claude_paper2_sealed_audit.py)
#   - Prints its own SHA-256 at startup; must equal the published seal.
#   - All randomness seeded (train seed 7777; audit inits generator 314159,
#     identical to the EquiPhase sealed audit).
#   - Training divergence (NaN/Inf loss) is itself a reported outcome, never
#     silently repaired: the epoch is skipped, counted, and printed.
#   - No thresholds are applied to baseline metrics; everything is reported
#     as measured, per the frozen expectation declarations.
# =============================================================================

import hashlib
import os
import platform
import sys
import time

import numpy as np
import torch
import torch.nn as nn

SCRIPT_VERSION = "v1.0-2026-08-08-claude-baselines-seal"
REPO_DIR = r"C:/Project/EquiPhase"
VANILLA_CKPT = REPO_DIR + r"/vanilla_deq_seed7777.pt"
MONOTONE_CKPT = REPO_DIR + r"/monotone_deq_seed7777.pt"

LATENT = 32
DT = 0.10
ETA = 0.20
TRAIN_SEED = 7777
EPOCHS = 50
TRAIN_SOLVER_STEPS = 100      # unrolled, matching the EquiPhase arm (disclosed
                              # deviation from IFT applies to all arms equally)
AUDIT_STEPS = 600
N_TRAJ = 100
SEALED_INIT_SEED = 314159
POINT_SEEDS = [101, 202, 303]
DIVERGE_NORM = 1e4
SEP = "=" * 88


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------- Baseline 1: Vanilla ---------------------------
class VanillaForceDEQ(nn.Module):
    """Unconstrained force field F(q,x): MLP 34 -> 64 -> 32.

    Same damped velocity Verlet integrator as EquiPhase, but the force is a
    free vector field (NOT the gradient of any potential)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(LATENT + 2, 64)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(64, LATENT)
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.zeros_(self.fc2.bias)

    def force(self, q, x):
        return self.fc2(self.act(self.fc1(torch.cat([q, x], dim=-1))))

    def step_batch(self, z, x):
        q, p = z[:, :LATENT], z[:, LATENT:]
        f1 = self.force(q, x)
        p_half = p + (DT / 2.0) * f1
        q_next = q + DT * p_half
        f2 = self.force(q_next, x)
        p_next = (1.0 - ETA) * (p_half + (DT / 2.0) * f2)
        return torch.cat([q_next, p_next], dim=-1)

    def step_single(self, z, x_single):
        return self.step_batch(z.unsqueeze(0), x_single.unsqueeze(0)).squeeze(0)


# ---------------------------- Baseline 2: Monotone ---------------------------
class MonotoneDEQ(nn.Module):
    """Contraction map z' = tanh(W z + U x + b) with ||W||_2 <= 0.9.

    By the Banach fixed-point theorem (tanh is 1-Lipschitz, ||W||_2 < 1) the
    map is a contraction on R^64 and has exactly ONE fixed point, independent
    of z_0. N_basins = 1 is therefore a STRUCTURAL consequence, not a finding."""

    W_SCALE = 0.9

    def __init__(self):
        super().__init__()
        self.W = nn.Linear(2 * LATENT, 2 * LATENT, bias=False)
        self.U = nn.Linear(2, 2 * LATENT, bias=True)
        nn.init.normal_(self.W.weight, std=0.05)
        nn.init.normal_(self.U.weight, std=0.05)
        nn.init.zeros_(self.U.bias)

    def _w_normed(self):
        w = self.W.weight
        sigma = torch.linalg.matrix_norm(w, ord=2)
        return w * (self.W_SCALE / torch.clamp(sigma, min=self.W_SCALE))

    def step_batch(self, z, x):
        return torch.tanh(z @ self._w_normed().t() + self.U(x))

    def step_single(self, z, x_single):
        return self.step_batch(z.unsqueeze(0), x_single.unsqueeze(0)).squeeze(0)


# ------------------------------- shared helpers ------------------------------
def solve(model, z0, x, steps):
    z = z0
    for _ in range(steps):
        z = model.step_batch(z, x)
    return z


def train_model(model, tag):
    print(f"\n--- TRAIN {tag} (seed {TRAIN_SEED}, {EPOCHS} epochs, "
          f"{TRAIN_SOLVER_STEPS}-step unrolled solver) ---")
    torch.manual_seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    dev = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bsz, half = 32, 16
    alphas = torch.rand(bsz, 1, device=dev) * 0.4 + 0.8
    xb = torch.cat([alphas, torch.sqrt(alphas)], dim=-1)
    skipped = 0
    for ep in range(1, EPOCHS + 1):
        opt.zero_grad()
        z0 = torch.randn(bsz, 2 * LATENT, device=dev) * 0.5
        z0[:half, 0] = torch.abs(z0[:half, 0]) + 0.5
        z0[half:, 0] = -torch.abs(z0[half:, 0]) - 0.5
        tq = torch.zeros(bsz, LATENT, device=dev)
        tq[:, 0:1] = torch.sign(z0[:, 0:1]) * torch.sqrt(alphas)
        z_star = solve(model, z0, xb, TRAIN_SOLVER_STEPS)
        loss_eq = torch.mean((z_star[:, :LATENT] - tq) ** 2)
        z_next = solve(model, z_star, xb, 1)
        loss_res = torch.mean((z_next - z_star) ** 2)
        total = loss_eq + 10.0 * loss_res
        if torch.isnan(total) or torch.isinf(total):
            skipped += 1
            if ep in (1, 10, 20, 30, 40, 50):
                print(f"  epoch {ep:2d}/{EPOCHS} | loss = NaN/Inf -> "
                      f"step SKIPPED (divergence is a reported outcome)")
            continue
        total.backward()
        opt.step()
        if ep in (1, 10, 20, 30, 40, 50):
            print(f"  epoch {ep:2d}/{EPOCHS} | loss_eq={loss_eq.item():.6e} "
                  f"| loss_res={loss_res.item():.6e}")
    print(f"  skipped (NaN/Inf) epochs: {skipped}/{EPOCHS}")
    return skipped


def measure_force_asymmetry(model, x_single, seed, dev):
    """Vanilla only: relative anti-symmetry of J_F at a seeded point (%)."""
    torch.manual_seed(seed)
    q_test = torch.randn(LATENT, device=dev)

    def f_fn(q_in):
        return model.force(q_in.unsqueeze(0), x_single.unsqueeze(0)).squeeze(0)

    J = torch.autograd.functional.jacobian(f_fn, q_test)
    asym = torch.linalg.norm(J - J.t(), ord="fro").item()
    nrm = torch.linalg.norm(J, ord="fro").item()
    return asym, nrm, (asym / (nrm + 1e-12)) * 100.0


def measure_c_R(model, x_single, seed, dev):
    torch.manual_seed(seed)
    z_test = torch.randn(2 * LATENT, device=dev)
    eye = torch.eye(LATENT, device=dev)
    zer = torch.zeros(LATENT, LATENT, device=dev)
    omega = torch.cat([torch.cat([zer, eye], 1), torch.cat([-eye, zer], 1)], 0)
    J = torch.autograd.functional.jacobian(
        lambda z: model.step_single(z, x_single), z_test)
    m = J.t() @ omega @ J
    c = (torch.trace(m @ omega.t()) / torch.trace(omega @ omega.t())).item()
    resid = torch.linalg.norm(m - c * omega, ord="fro").item()
    tgt = torch.linalg.norm(c * omega, ord="fro").item()
    return c, resid / (abs(tgt) + 1e-12)


def audit_trajectories(model, x_single, z_inits, dev, tag):
    rows = []
    for i in range(z_inits.shape[0]):
        z = z_inits[i].to(dev)
        q1_init = z[0].item()
        diverged = False
        for _ in range(AUDIT_STEPS):
            z = model.step_single(z, x_single)
            if (torch.isnan(z).any() or torch.isinf(z).any()
                    or torch.norm(z) > DIVERGE_NORM):
                diverged = True
                break
        if diverged:
            rows.append((i, q1_init, "DIVERGED", float("nan"), float("nan"),
                         float("nan"), float("nan")))
            continue
        z_det = z.detach()
        res = torch.norm(model.step_single(z_det, x_single) - z_det).item()
        J = torch.autograd.functional.jacobian(
            lambda zz: model.step_single(zz, x_single), z_det)
        ev = torch.linalg.eigvals(J.detach().to(torch.float64))
        rho = torch.max(torch.abs(ev)).item()
        rows.append((i, q1_init, "CONVERGED", z_det[0].item(), z_det[1].item(),
                     res, rho))

    print(f"\n[{tag}] per-trajectory table "
          f"(idx | q1_init | status | q1_fin | q2_fin | residual | exact_rho)")
    for r in rows:
        print(f"  {r[0]:3d} | {r[1]:+.6f} | {r[2]:>9s} | "
              f"{r[3]:+.6f} | {r[4]:+.6f} | {r[5]:.6e} | {r[6]:.6f}")

    conv = [r for r in rows if r[2] == "CONVERGED"]
    div = len(rows) - len(conv)
    print(f"\n[{tag}] summary: converged={len(conv)}  diverged={div}")
    if conv:
        ends = np.array([[r[3], r[4]] for r in conv])
        # greedy epsilon-clustering of endpoints (eps = 0.10) on (q1,q2)
        centers, counts = [], []
        for e in ends:
            for k, cpt in enumerate(centers):
                if np.linalg.norm(e - cpt) < 0.10:
                    counts[k] += 1
                    break
            else:
                centers.append(e.copy())
                counts.append(1)
        print(f"  N_endpoint_clusters (eps=0.10 on (q1,q2)) = {len(centers)}")
        for k, (cpt, cnt) in enumerate(zip(centers, counts)):
            print(f"    cluster {k}: center=({cpt[0]:+.6f},{cpt[1]:+.6f}) "
                  f"members={cnt}")
        rhos = np.array([r[6] for r in conv])
        resids = np.array([r[5] for r in conv])
        print(f"  residual: mean={resids.mean():.6e} max={resids.max():.6e}")
        print(f"  exact rho: mean={rhos.mean():.6f} min={rhos.min():.6f} "
              f"max={rhos.max():.6f}")
    return rows


def main():
    t0 = time.time()
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(SEP)
    print("=== CLAUDE SEALED BASELINE AUDIT (Paper 2 Step 12: Vanilla / Monotone) ===")
    print(SEP)
    print(f"  script_version = {SCRIPT_VERSION}")
    print(f"  script_sha256  = {sha256_file(os.path.abspath(__file__))}")
    print(f"  python = {sys.version.split()[0]} | torch = {torch.__version__} "
          f"| device = {dev} | platform = {platform.platform()}")

    gen = torch.Generator(device="cpu").manual_seed(SEALED_INIT_SEED)
    z_sealed = torch.randn(N_TRAJ, 2 * LATENT, generator=gen) * 2.0
    z_hash = hashlib.sha256(z_sealed.numpy().tobytes()).hexdigest()
    print(f"  sealed inits: generator seed {SEALED_INIT_SEED} | "
          f"array sha256 = {z_hash}")

    x_a10 = torch.tensor([1.0, 1.0], device=dev)

    # ---------------- Baseline 1: Vanilla ----------------
    print("\n" + SEP)
    print("=== BASELINE 1: VANILLA DEQ (unconstrained force MLP 34->64->32) ===")
    print(SEP)
    van = VanillaForceDEQ().to(dev)
    n_par = sum(p.numel() for p in van.parameters())
    print(f"  parameter count = {n_par}")
    train_model(van, "VANILLA")
    torch.save(van.state_dict(), VANILLA_CKPT)
    print(f"  checkpoint saved: {VANILLA_CKPT}")
    print(f"  checkpoint sha256 = {sha256_file(VANILLA_CKPT)}")

    print("\n[VANILLA] force anti-symmetry (measured, no prediction):")
    for s in POINT_SEEDS:
        a, n, pct = measure_force_asymmetry(van, x_a10, s, dev)
        print(f"  seed {s}: asym_norm = {a:.6e} | norm_JF = {n:.6e} "
              f"| ratio = {pct:.4f} %")
    print("  [reference] random 32x32 Gaussian baseline: 139.2% +- 3.2%")
    print("\n[VANILLA] conformal residual (measured post-hoc, no prediction):")
    for s in POINT_SEEDS:
        c, r = measure_c_R(van, x_a10, s, dev)
        print(f"  seed {s}: c = {c:.6f} | R = {r:.6e}")
    audit_trajectories(van, x_a10, z_sealed, dev, "VANILLA-sealed")

    # ---------------- Baseline 2: Monotone ----------------
    print("\n" + SEP)
    print("=== BASELINE 2: MONOTONE DEQ (contraction, ||W||_2 <= 0.9) ===")
    print(SEP)
    mono = MonotoneDEQ().to(dev)
    n_par = sum(p.numel() for p in mono.parameters())
    print(f"  parameter count = {n_par}")
    sigma = torch.linalg.matrix_norm(mono._w_normed(), ord=2).item()
    print(f"  effective ||W||_2 after normalization = {sigma:.6f} (<= 0.9)")
    train_model(mono, "MONOTONE")
    torch.save(mono.state_dict(), MONOTONE_CKPT)
    print(f"  checkpoint saved: {MONOTONE_CKPT}")
    print(f"  checkpoint sha256 = {sha256_file(MONOTONE_CKPT)}")
    sigma = torch.linalg.matrix_norm(mono._w_normed(), ord=2).item()
    print(f"  effective ||W||_2 post-training = {sigma:.6f} (<= 0.9)")
    audit_trajectories(mono, x_a10, z_sealed, dev, "MONOTONE-sealed")
    print("\n  [structural note] N_basins = 1 for the Monotone arm is a "
          "consequence of the Banach fixed-point theorem, NOT an empirical "
          "finding. The audit above merely illustrates it.")

    print(f"\n[END] TOTAL WALL-CLOCK = {time.time() - t0:.1f} s")
    print(SEP)
    print("=== END OF CLAUDE SEALED BASELINE STDOUT ===")
    print(SEP)


if __name__ == "__main__":
    main()
