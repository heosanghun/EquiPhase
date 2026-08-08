# =============================================================================
# claude_paper2_sealed_audit.py
# Author  : Claude (external auditor, System 2)  --  SEALED SCRIPT, DO NOT MODIFY
# Version : v1.0 (2026-08-08)
# Purpose : Single-pass deterministic verification of the seed-7777 checkpoint
#           for Paper 2 (EquiPhase / MS-DEQ). Produces canonical stdout only.
#
# INTEGRITY MODEL
#   1. This script prints its own SHA-256 at startup. It must equal the seal
#      hash published by Claude in the audit conversation. Any mismatch means
#      the script was modified -> the entire run is VOID.
#   2. The checkpoint SHA-256 is verified before loading. Mismatch -> ABORT.
#   3. All randomness is seeded. Two consecutive runs on the same machine must
#      agree to the last printed digit (CPU sections exactly; CUDA sections up
#      to library nondeterminism, which is why the trajectory section is also
#      run once on CPU with sealed inits).
#
# USAGE
#   python claude_paper2_sealed_audit.py            (verification only)
#   python claude_paper2_sealed_audit.py --retrain  (adds wall-clock retrain,
#                                                    saves to *_retrained.pt,
#                                                    NEVER overwrites original)
# =============================================================================

import argparse
import csv
import hashlib
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# ----------------------------- sealed constants ------------------------------
SCRIPT_VERSION = "v1.0-2026-08-08-claude-seal"
EXPECTED_CKPT_SHA256 = "c6b64ec3961f31bc8c9758f80b91b9168ed2d06d9df0a07c3b6989f8f699ba97"
REPO_DIR = r"C:/Project/EquiPhase"
CKPT_PATH = REPO_DIR + r"/supervised_deq_model_seed7777.pt"
CSV_PATH = REPO_DIR + r"/trajectory_basins_seed7777.csv"
ZINITS_PATH = REPO_DIR + r"/z_inits_sealed.pt"
RETRAIN_PATH = REPO_DIR + r"/supervised_deq_model_seed7777_retrained.pt"

LATENT = 32
DT = 0.10
ETA = 0.20
TRAJ_STEPS = 600
N_TRAJ = 100
SEALED_INIT_SEED = 314159  # Claude-chosen seed for the fresh sealed inits
POINT_SEEDS = [101, 202, 303]  # fixed probe points for G1 / G2

SEP = "=" * 88


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_stat_line(path):
    if not os.path.exists(path):
        return f"  {path} : MISSING"
    st = os.stat(path)
    return (f"  {path}\n"
            f"    sha256 = {sha256_file(path)}\n"
            f"    size   = {st.st_size} bytes | mtime = "
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime))}")


def run_git(args):
    try:
        out = subprocess.run(["git"] + args, cwd=REPO_DIR, capture_output=True,
                             text=True, timeout=60)
        return (out.stdout + out.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"<git unavailable: {e}>"


# ------------------------- model (replicated verbatim) -----------------------
# Architecture replicated from train_paper2_deq_supervised.py as submitted.
# state_dict keys: fc1.weight, fc1.bias, fc2.weight, fc2.bias
class EquiPhaseSupervisedDEQ(nn.Module):
    def __init__(self, latent_dim=LATENT, damping=ETA, dt=DT):
        super().__init__()
        self.latent_dim = latent_dim
        self.damping = damping
        self.dt = dt
        self.fc1 = nn.Linear(latent_dim + 2, 64)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(64, 1)

    def v_base(self, q, x):
        alpha = x[:, 0:1]
        q_sq = torch.sum(q ** 2, dim=-1, keepdim=True)
        q_a_q = (alpha * (q[:, 0:1] ** 2) + 0.3 * (q[:, 1:2] ** 2)
                 + torch.sum(-0.5 * (q[:, 2:] ** 2), dim=-1, keepdim=True))
        return 0.25 * (q_sq ** 2) - 0.5 * q_a_q

    def v_net(self, q, x):
        qx = torch.cat([q, x], dim=-1)
        return self.fc2(self.act(self.fc1(qx)))

    def V_total(self, q, x):
        return self.v_base(q, x) + self.v_net(q, x)

    def grad_V(self, q, x):
        q_req = q if q.requires_grad else q.detach().requires_grad_(True)
        v = self.V_total(q_req, x)
        return torch.autograd.grad(v.sum(), q_req, create_graph=True,
                                   retain_graph=True)[0]

    # damped velocity-Verlet cell, replicated verbatim from the training script
    def cell_forward_single(self, z, x_single):
        q = z[:self.latent_dim].unsqueeze(0)
        p = z[self.latent_dim:].unsqueeze(0)
        x = x_single.unsqueeze(0)
        g1 = self.grad_V(q, x)
        p_half = p - (self.dt / 2.0) * g1
        q_next = q + self.dt * p_half
        g2 = self.grad_V(q_next, x)
        p_uncut = p_half - (self.dt / 2.0) * g2
        p_next = (1.0 - self.damping) * p_uncut
        return torch.cat([q_next, p_next], dim=-1).squeeze(0)


# ------------------------------- gate helpers --------------------------------
def measure_g1(model, x_single, seed, device):
    torch.manual_seed(seed)
    q_test = torch.randn(LATENT, device=device)

    def force_fn(q_in):
        return -model.grad_V(q_in.unsqueeze(0), x_single.unsqueeze(0)).squeeze(0)

    J = torch.autograd.functional.jacobian(force_fn, q_test)
    asym = torch.linalg.norm(J - J.t(), ord="fro").item()
    nrm = torch.linalg.norm(J, ord="fro").item()
    return asym, nrm, (asym / (nrm + 1e-12)) * 100.0


def measure_g2(model, x_single, seed, device):
    torch.manual_seed(seed)
    z_test = torch.randn(2 * LATENT, device=device)
    half = LATENT
    eye = torch.eye(half, device=device)
    zer = torch.zeros(half, half, device=device)
    omega = torch.cat([torch.cat([zer, eye], 1), torch.cat([-eye, zer], 1)], 0)

    def f_map(z):
        return model.cell_forward_single(z, x_single)

    J = torch.autograd.functional.jacobian(f_map, z_test)
    m = J.t() @ omega @ J
    c = (torch.trace(m @ omega.t()) / torch.trace(omega @ omega.t())).item()
    resid = torch.linalg.norm(m - c * omega, ord="fro").item()
    tgt = torch.linalg.norm(c * omega, ord="fro").item()
    return c, resid / (tgt + 1e-12)


def exact_spectral_radius(model, x_single, z_point):
    def f_map(z):
        return model.cell_forward_single(z, x_single)

    J = torch.autograd.functional.jacobian(f_map, z_point)
    ev = torch.linalg.eigvals(J.detach().to(torch.float64))
    return torch.max(torch.abs(ev)).item()


def run_trajectories(model, x_single, z_inits, device, tag):
    rows = []
    for i in range(z_inits.shape[0]):
        z = z_inits[i].to(device)
        q1_init = z[0].item()
        diverged = False
        for _ in range(TRAJ_STEPS):
            z = model.cell_forward_single(z, x_single)
            if (torch.isnan(z).any() or torch.isinf(z).any()
                    or torch.norm(z) > 1e4):
                diverged = True
                break
        if diverged:
            rows.append((i, q1_init, "DIVERGED", float("nan"), float("nan"),
                         float("nan"), float("nan")))
            continue
        z_det = z.detach()
        z_next = model.cell_forward_single(z_det, x_single)
        res = torch.norm(z_next - z_det).item()
        rho = exact_spectral_radius(model, x_single, z_det)
        rows.append((i, q1_init, "CONVERGED", z_det[0].item(), z_det[1].item(),
                     res, rho))

    print(f"\n[{tag}] per-trajectory table "
          f"(idx | q1_init | status | q1_fin | q2_fin | residual | exact_rho)")
    for r in rows:
        print(f"  {r[0]:3d} | {r[1]:+.6f} | {r[2]:>9s} | "
              f"{r[3]:+.6f} | {r[4]:+.6f} | {r[5]:.6e} | {r[6]:.6f}")

    conv = [r for r in rows if r[2] == "CONVERGED"]
    plus = [r for r in conv if r[3] > 0.1]
    minus = [r for r in conv if r[3] < -0.1]
    spur = len(conv) - len(plus) - len(minus)
    residuals = np.array([r[5] for r in conv]) if conv else np.array([np.nan])
    rhos = np.array([r[6] for r in conv]) if conv else np.array([np.nan])

    ct = {"p2p": sum(1 for r in conv if r[1] > 0 and r[3] > 0.1),
          "p2m": sum(1 for r in conv if r[1] > 0 and r[3] < -0.1),
          "m2p": sum(1 for r in conv if r[1] < 0 and r[3] > 0.1),
          "m2m": sum(1 for r in conv if r[1] < 0 and r[3] < -0.1)}

    print(f"\n[{tag}] summary")
    print(f"  converged={len(conv)}  diverged={N_TRAJ - len(conv)}  "
          f"plus={len(plus)}  minus={len(minus)}  spurious={spur}")
    print(f"  dominant_share = {(len(plus) + len(minus)) / N_TRAJ:.4f}")
    print(f"  G3' residual: mean={residuals.mean():.6e}  "
          f"max={residuals.max():.6e}")
    print(f"  exact rho:    mean={rhos.mean():.6f}  min={rhos.min():.6f}  "
          f"max={rhos.max():.6f}")
    print(f"  [theory reference] complex-pair modes imply "
          f"|lambda| = sqrt(1-eta) = {np.sqrt(1.0 - ETA):.6f}")
    print(f"  cross-tab init-sign vs basin: "
          f"q1(0)>0->plus {ct['p2p']} | q1(0)>0->minus {ct['p2m']} | "
          f"q1(0)<0->plus {ct['m2p']} | q1(0)<0->minus {ct['m2m']}")
    print(f"  SetA cross-tab was 45/6/0/49 ; SetB was 16/37/29/18 "
          f"-> this run matches: "
          f"{'SetA' if (ct['p2p'], ct['p2m'], ct['m2p'], ct['m2m']) == (45, 6, 0, 49) else 'SetB' if (ct['p2p'], ct['p2m'], ct['m2p'], ct['m2m']) == (16, 37, 29, 18) else 'NEITHER (third value)'}")
    return rows


def newton_critical_point(model, x_single, q_init, device, iters=60, tol=1e-9):
    q = q_init.clone().detach().to(device).requires_grad_(True)
    x_in = x_single.unsqueeze(0)
    grad_norm = float("nan")
    for _ in range(iters):
        g = model.grad_V(q.unsqueeze(0), x_in).squeeze(0)
        grad_norm = torch.norm(g).item()
        if grad_norm < tol:
            break
        H = torch.autograd.functional.hessian(
            lambda qq: model.V_total(qq.unsqueeze(0), x_in).sum(), q.detach())
        step = torch.linalg.solve(H.to(torch.float64),
                                  g.detach().to(torch.float64).unsqueeze(-1))
        q = (q.detach() - step.squeeze(-1).to(q.dtype)).requires_grad_(True)
    return q.detach(), grad_norm


def grad_vnet_norm(model, q_point, x_single):
    q = q_point.clone().detach().requires_grad_(True)
    v = model.v_net(q.unsqueeze(0), x_single.unsqueeze(0))
    g = torch.autograd.grad(v.sum(), q)[0]
    return torch.norm(g).item(), v.item()


# ------------------------------------ main -----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    print(SEP)
    print("=== CLAUDE SEALED AUDIT (Paper 2 / seed-7777 checkpoint) ===")
    print(SEP)

    # SEC 0 -- environment & integrity
    print("\n[SEC 0] ENVIRONMENT & INTEGRITY")
    print(f"  script_version = {SCRIPT_VERSION}")
    print(f"  script_path    = {os.path.abspath(__file__)}")
    print(f"  script_sha256  = {sha256_file(os.path.abspath(__file__))}")
    print(f"  python  = {sys.version.split()[0]} | torch = {torch.__version__}"
          f" | cuda_available = {torch.cuda.is_available()}")
    print(f"  platform = {platform.platform()}")
    dev_gpu = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dev_cpu = torch.device("cpu")
    print(f"  primary device = {dev_gpu}")
    print("  file inventory:")
    for p in [CKPT_PATH, CSV_PATH,
              REPO_DIR + r"/train_paper2_deq_supervised.py",
              REPO_DIR + r"/run_paper2_rigorous_gates.py",
              REPO_DIR + r"/run_raw_stdout_audit.py",
              REPO_DIR + r"/prereg_1d_double_well.md"]:
        print(file_stat_line(p))
    print("  git log --graph --format='%H %ci %s' -n 12 :")
    print("    " + run_git(["log", "--graph", "--format=%H %ci %s",
                            "-n", "12"]).replace("\n", "\n    "))
    for commit in ["97cd2b5", "e148866"]:
        print(f"  git show --stat {commit} :")
        print("    " + run_git(["show", "--stat", "--format=%H %ci %s",
                                commit]).replace("\n", "\n    "))

    # SEC 1 -- checkpoint gate
    print("\n[SEC 1] CHECKPOINT INTEGRITY GATE")
    actual = sha256_file(CKPT_PATH)
    print(f"  expected sha256 = {EXPECTED_CKPT_SHA256}")
    print(f"  actual   sha256 = {actual}")
    if actual != EXPECTED_CKPT_SHA256:
        print("  RESULT: MISMATCH -> ABORTING. Entire run VOID.")
        sys.exit(1)
    print("  RESULT: MATCH -> proceeding.")

    state = torch.load(CKPT_PATH, map_location=dev_gpu)
    print(f"  state_dict keys = {sorted(state.keys())}")
    model = EquiPhaseSupervisedDEQ().to(dev_gpu)
    model.load_state_dict(state, strict=True)
    model.eval()
    x_a10 = torch.tensor([1.0, 1.0], device=dev_gpu)

    # SEC 2 -- G1 measured (never hardcoded)
    print("\n[SEC 2] G1 FORCE ANTI-SYMMETRY (measured at 3 seeded points)")
    for s in POINT_SEEDS:
        asym, nrm, pct = measure_g1(model, x_a10, s, dev_gpu)
        print(f"  seed {s}: asym_norm = {asym:.10e} | norm_JF = {nrm:.10e} "
              f"| ratio = {pct:.6e} %")
    print("  SetA raw was (0.0 / 11.568) ; SetB raw was (1.646e-10 / 127.56)")

    # SEC 3 -- G2 conformal symplectic residual
    print("\n[SEC 3] G2 CONFORMAL SYMPLECTIC (measured at 3 seeded points)")
    for s in POINT_SEEDS:
        c, r = measure_g2(model, x_a10, s, dev_gpu)
        print(f"  seed {s}: c = {c:.8f} | R = {r:.10e}")
    print(f"  [theory] damped Verlet cell satisfies J^T Omega J = (1-eta) Omega"
          f" identically -> c = {1.0 - ETA:.8f}, R at float32 eps")

    # SEC 4 -- replicate the original audit inits (seeds 9000+i) on primary dev
    print("\n[SEC 4] TRAJECTORY AUDIT - ORIGINAL SCHEME (seeds 9000+i, "
          f"device={dev_gpu})")
    inits_orig = []
    for i in range(N_TRAJ):
        torch.manual_seed(9000 + i)
        inits_orig.append(torch.randn(2 * LATENT, device=dev_gpu) * 2.0)
    inits_orig = torch.stack(inits_orig)
    rows_orig = run_trajectories(model, x_a10, inits_orig, dev_gpu,
                                 "SEC4-origscheme")

    # CSV comparison
    print("\n[SEC 4b] COMPARISON AGAINST trajectory_basins_seed7777.csv")
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="") as f:
            csv_rows = list(csv.DictReader(f))
        n_match = 0
        n_cmp = min(len(csv_rows), len(rows_orig))
        for i in range(n_cmp):
            try:
                q1c = float(csv_rows[i]["q1_final"])
                q2c = float(csv_rows[i]["q2_final"])
            except (ValueError, KeyError):
                continue
            q1r, q2r = rows_orig[i][3], rows_orig[i][4]
            ok = (abs(q1c - q1r) < 1e-3 and abs(q2c - q2r) < 1e-3)
            n_match += int(ok)
            if i < 10 or not ok:
                print(f"  row {i:3d}: csv=({q1c:+.6f},{q2c:+.6f}) "
                      f"rerun=({q1r:+.6f},{q2r:+.6f}) -> "
                      f"{'MATCH' if ok else 'MISMATCH'}")
        print(f"  endpoint match: {n_match}/{n_cmp} rows within 1e-3")
    else:
        print("  CSV missing -> comparison skipped")

    # SEC 5 -- fresh sealed inits on CPU (bit-reproducible across runs)
    print(f"\n[SEC 5] TRAJECTORY AUDIT - SEALED FRESH INITS "
          f"(generator seed {SEALED_INIT_SEED}, device=cpu)")
    gen = torch.Generator(device="cpu").manual_seed(SEALED_INIT_SEED)
    z_sealed = torch.randn(N_TRAJ, 2 * LATENT, generator=gen) * 2.0
    torch.save(z_sealed, ZINITS_PATH)
    print(f"  z_inits_sealed.pt saved | sha256 = {sha256_file(ZINITS_PATH)}")
    model_cpu = EquiPhaseSupervisedDEQ().to(dev_cpu)
    model_cpu.load_state_dict(
        {k: v.to(dev_cpu) for k, v in state.items()}, strict=True)
    model_cpu.eval()
    x_cpu = torch.tensor([1.0, 1.0])
    run_trajectories(model_cpu, x_cpu, z_sealed, dev_cpu, "SEC5-sealed-cpu")

    # SEC 6 -- Newton critical points per alpha
    print("\n[SEC 6] NEWTON CRITICAL POINTS (per alpha)")
    for alpha in [0.8, 1.0, 1.2]:
        x_a = torch.tensor([alpha, float(np.sqrt(alpha))], device=dev_gpu)
        q0_min = torch.zeros(LATENT, device=dev_gpu)
        q0_min[0] = float(np.sqrt(alpha))
        q_min, gmin = newton_critical_point(model, x_a, q0_min, dev_gpu)
        ref_min = torch.zeros(LATENT, device=dev_gpu)
        ref_min[0] = float(np.sqrt(alpha))
        d_min = torch.norm(q_min - ref_min).item()
        lim = 1e-2 / (2 * alpha)
        print(f"  alpha={alpha}: min  q1={q_min[0].item():+.6f} "
              f"q2={q_min[1].item():+.6f} ||q_rest||="
              f"{torch.norm(q_min[2:]).item():.6f} | ||grad||={gmin:.3e} "
              f"| disp={d_min:.6f} (per-alpha lim {lim:.6f}; sealed 6.25e-3)")

        q0_sad = torch.zeros(LATENT, device=dev_gpu)
        q0_sad[1] = float(np.sqrt(0.3))
        q_sad, gsad = newton_critical_point(model, x_a, q0_sad, dev_gpu)
        ref_sad = torch.zeros(LATENT, device=dev_gpu)
        ref_sad[1] = float(np.sqrt(0.3))
        d_sad = torch.norm(q_sad - ref_sad).item()
        print(f"  alpha={alpha}: sad  q1={q_sad[0].item():+.6f} "
              f"q2={q_sad[1].item():+.6f} ||q_rest||="
              f"{torch.norm(q_sad[2:]).item():.6f} | ||grad||={gsad:.3e} "
              f"| disp={d_sad:.6f} (lim 1.67e-2)")

        if abs(alpha - 1.0) < 1e-9:
            H = torch.autograd.functional.hessian(
                lambda qq: model.V_total(qq.unsqueeze(0),
                                         x_a.unsqueeze(0)).sum(),
                q_sad)
            ev = torch.linalg.eigvalsh(
                0.5 * (H + H.t()).to(torch.float64)).cpu().numpy()
            print("  alpha=1.0 saddle Hessian eigenvalues (all 32, ascending):")
            for k, lam in enumerate(ev, 1):
                print(f"    lambda_{k:02d} = {lam:+.6f}")
            v_min = model.V_total(q_min.unsqueeze(0), x_a.unsqueeze(0)).item()
            v_sad = model.V_total(q_sad.unsqueeze(0), x_a.unsqueeze(0)).item()
            print(f"  [G7'] V(min)={v_min:+.6f} V(saddle)={v_sad:+.6f} "
                  f"deltaV={v_sad - v_min:+.6f} (target 0.2275 +- 0.0100)")

            gn_min, vv_min = grad_vnet_norm(model, q_min, x_a)
            gn_sad, vv_sad = grad_vnet_norm(model, q_sad, x_a)
            print(f"  [v_net] at min:    ||grad v_net|| = {gn_min:.6e} "
                  f"| v_net = {vv_min:+.6e}")
            print(f"  [v_net] at saddle: ||grad v_net|| = {gn_sad:.6e} "
                  f"| v_net = {vv_sad:+.6e}")
            print("  [prereg assumption] ||grad eps|| <= 1.0e-2 -> "
                  f"{'HOLDS' if max(gn_min, gn_sad) <= 1e-2 else 'VIOLATED'} "
                  "at the two critical points")

    # SEC 7 -- optional wall-clock retrain
    if args.retrain:
        print("\n[SEC 7] WALL-CLOCK RETRAIN (seed 7777, 50 epochs, "
              "saved to *_retrained.pt)")
        t_r = time.time()
        torch.manual_seed(7777)
        np.random.seed(7777)
        m2 = EquiPhaseSupervisedDEQ().to(dev_gpu)
        opt = torch.optim.Adam(m2.parameters(), lr=1e-3)
        bsz, half = 32, 16
        alphas = torch.rand(bsz, 1, device=dev_gpu) * 0.4 + 0.8
        xb = torch.cat([alphas, torch.sqrt(alphas)], dim=-1)
        for ep in range(1, 51):
            opt.zero_grad()
            z0 = torch.randn(bsz, 64, device=dev_gpu) * 0.5
            z0[:half, 0] = torch.abs(z0[:half, 0]) + 0.5
            z0[half:, 0] = -torch.abs(z0[half:, 0]) - 0.5
            tq = torch.zeros(bsz, LATENT, device=dev_gpu)
            tq[:, 0:1] = torch.sign(z0[:, 0:1]) * torch.sqrt(alphas)
            zc = z0
            for _ in range(100):
                q = zc[:, :LATENT]
                p = zc[:, LATENT:]
                g1 = m2.grad_V(q, xb)
                ph = p - (DT / 2.0) * g1
                qn = q + DT * ph
                g2 = m2.grad_V(qn, xb)
                pn = (1.0 - ETA) * (ph - (DT / 2.0) * g2)
                zc = torch.cat([qn, pn], dim=-1)
            loss_eq = torch.mean((zc[:, :LATENT] - tq) ** 2)
            q = zc[:, :LATENT]
            p = zc[:, LATENT:]
            g1 = m2.grad_V(q, xb)
            ph = p - (DT / 2.0) * g1
            qn = q + DT * ph
            g2 = m2.grad_V(qn, xb)
            pn = (1.0 - ETA) * (ph - (DT / 2.0) * g2)
            zn = torch.cat([qn, pn], dim=-1)
            loss_res = torch.mean((zn - zc) ** 2)
            (loss_eq + 10.0 * loss_res).backward()
            opt.step()
            if ep in (1, 10, 20, 30, 40, 50):
                print(f"  epoch {ep:2d}/50 | loss_eq={loss_eq.item():.6e} "
                      f"| loss_res={loss_res.item():.6e}")
        torch.save(m2.state_dict(), RETRAIN_PATH)
        print(f"  retrained ckpt sha256 = {sha256_file(RETRAIN_PATH)} "
              f"(original = {EXPECTED_CKPT_SHA256[:16]}...)")
        print(f"  RETRAIN WALL-CLOCK = {time.time() - t_r:.1f} s "
              f"(reported original window was ~257 s incl. audit+commit)")

    print(f"\n[END] TOTAL WALL-CLOCK = {time.time() - t0:.1f} s")
    print(SEP)
    print("=== END OF CLAUDE SEALED AUDIT STDOUT ===")
    print(SEP)


if __name__ == "__main__":
    main()
