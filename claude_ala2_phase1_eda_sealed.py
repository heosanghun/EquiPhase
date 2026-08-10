# -*- coding: utf-8 -*-
# =============================================================================
# claude_ala2_phase1_eda_sealed.py
# SEALED AUDIT SCRIPT — authored solely by external auditor (Claude, System 2)
# Purpose : Phase 1 EDA for alanine dipeptide (mdshare) backbone dihedrals.
#           Deterministic detection of metastable states (density modes) in
#           (phi, psi) space, to fix R2/R3 gate reference values from DATA,
#           not from vacuum-phase literature.
# Rules   : Direct invocation only. No modification. Submit stdout verbatim.
#           Any hash mismatch aborts the run (fail-closed).
# Usage   : python claude_ala2_phase1_eda_sealed.py
#           (run from repository root /home/user/EquiPhase)
# =============================================================================
import hashlib
import math
import os
import sys

import numpy as np

DATA_PATH = os.path.join("data", "ala2",
                         "alanine-dipeptide-3x250ns-backbone-dihedrals.npz")
# Data anchor sealed on 2026-08-08 (auditor ledger, Phase 0/1 report P2)
ANCHOR_SHA256 = "F5AD30768A7CF3451B3061CB2ECB7F7D1DE8C13044534376D26A6653D4CD5717"

NBINS = 72                  # 5-degree bins over [-pi, pi)
MIN_MODE_FRACTION = 0.001   # a mode must hold >= 0.1% of all samples in its bin
TOP_K = 6                   # report at most 6 modes
BOX_DEG = 20.0              # population box half-width around each mode (deg)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def fail(msg):
    print("EDA_ABORT: " + msg)
    print("EDA_END")
    sys.exit(1)


def main():
    print("EDA_BEGIN")
    print("script=claude_ala2_phase1_eda_sealed.py")
    print("numpy=" + np.__version__)

    if not os.path.exists(DATA_PATH):
        fail("data file not found: " + DATA_PATH)

    digest = sha256_of(DATA_PATH)
    print("data_sha256=" + digest)
    if digest != ANCHOR_SHA256:
        fail("SHA-256 mismatch vs sealed anchor — refusing to proceed")
    print("anchor_check=PASS")

    npz = np.load(DATA_PATH)
    keys = sorted(npz.files)
    print("npz_keys=" + ",".join(keys))

    parts = []
    for k in keys:
        a = np.asarray(npz[k])
        if a.ndim != 2 or a.shape[1] != 2:
            fail("array %s has unexpected shape %s" % (k, a.shape))
        parts.append(a.astype(np.float64))
        print("array %s shape=%s" % (k, a.shape))
    X = np.concatenate(parts, axis=0)
    n = X.shape[0]
    print("total_samples=%d" % n)

    vmax = float(np.max(np.abs(X)))
    if vmax > math.pi + 0.01:
        fail("values exceed [-pi, pi]; units are not radians (max=%f)" % vmax)
    print("units_check=radians PASS (max_abs=%.6f)" % vmax)

    phi = X[:, 0]
    psi = X[:, 1]

    # ---- basic split diagnostics -------------------------------------------
    frac_phi_pos = float(np.mean(phi > 0.0))
    print("fraction_phi_positive=%.6f" % frac_phi_pos)

    # ---- 2D histogram with periodic local-maximum detection ----------------
    edges = np.linspace(-math.pi, math.pi, NBINS + 1)
    H, _, _ = np.histogram2d(phi, psi, bins=[edges, edges])
    H = H / float(n)

    def nb(i, d):
        return (i + d) % NBINS  # periodic wrap

    modes = []
    for i in range(NBINS):
        for j in range(NBINS):
            c = H[i, j]
            if c < MIN_MODE_FRACTION:
                continue
            is_max = True
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    if H[nb(i, di), nb(j, dj)] > c:
                        is_max = False
                        break
                if not is_max:
                    break
            if is_max:
                modes.append((c, i, j))
    modes.sort(reverse=True)
    modes = modes[:TOP_K]

    centers = 0.5 * (edges[:-1] + edges[1:])
    box = math.radians(BOX_DEG)

    print("modes_detected=%d (top %d by bin density, min_bin_fraction=%.4f)"
          % (len(modes), TOP_K, MIN_MODE_FRACTION))
    print("mode_table_columns=rank,phi_deg,psi_deg,bin_fraction,"
          "box_population_fraction(+-%.0fdeg),F_over_kT" % BOX_DEG)

    pmax = modes[0][0] if modes else float("nan")
    for r, (c, i, j) in enumerate(modes, start=1):
        cp, cq = centers[i], centers[j]
        dphi = np.abs(np.angle(np.exp(1j * (phi - cp))))
        dpsi = np.abs(np.angle(np.exp(1j * (psi - cq))))
        pop = float(np.mean((dphi <= box) & (dpsi <= box)))
        f_over_kt = -math.log(c / pmax) if c > 0 else float("inf")
        print("MODE %d | phi=%+8.2f deg | psi=%+8.2f deg | bin_frac=%.6f | "
              "box_pop=%.6f | F/kT_rel=%.4f"
              % (r, math.degrees(cp), math.degrees(cq), c, pop, f_over_kt))

    # ---- two-basin summary along phi (barrier proxy) -----------------------
    Hphi, _ = np.histogram(phi, bins=edges)
    Hphi = Hphi / float(n)
    # deterministic barrier proxy: minimum of 1D phi-density between the two
    # highest 1D phi modes
    i_sorted = np.argsort(Hphi)[::-1]
    i1 = int(i_sorted[0])
    i2 = None
    for cand in i_sorted[1:]:
        if min(abs(int(cand) - i1), NBINS - abs(int(cand) - i1)) >= 6:
            i2 = int(cand)
            break
    if i2 is not None:
        lo, hi = min(i1, i2), max(i1, i2)
        inner = Hphi[lo:hi + 1]
        outer = np.concatenate([Hphi[hi:], Hphi[:lo + 1]])
        path = inner if inner.min() >= outer.min() else outer
        pmin = float(path.min())
        p1, p2 = float(Hphi[i1]), float(Hphi[i2])
        print("phi_mode_1=%+8.2f deg (frac=%.6f)"
              % (math.degrees(centers[i1]), p1))
        print("phi_mode_2=%+8.2f deg (frac=%.6f)"
              % (math.degrees(centers[i2]), p2))
        if pmin > 0.0:
            print("barrier_proxy_F/kT: from_mode1=%.4f from_mode2=%.4f"
                  % (math.log(p1 / pmin), math.log(p2 / pmin)))
        else:
            print("barrier_proxy_F/kT: saddle bin empty at this resolution")
    else:
        print("phi 1D analysis: second separated mode not found")

    print("EDA_END")


if __name__ == "__main__":
    main()
