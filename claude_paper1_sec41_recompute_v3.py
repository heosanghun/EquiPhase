# =============================================================================
# claude_paper1_sec41_recompute_v3.py
# Author  : Claude (external auditor, System 2)  --  SEALED SCRIPT, DO NOT MODIFY
# Version : v3.0 (2026-08-08)
# Purpose : Final adjudication for Paper 1 Section 4.1 Task B strata.
#           v2 established that the original train exclusion was NOT the bare
#           cell "UNKNOWN" (0 such cells) and that 17 rows contain the word
#           'unknown' as free text or have empty parsed sequences, of which
#           exactly 15 are positives -- arithmetically matching the ledger
#           census (2554-15=2539, 1749-15=1734, neg 805 unchanged).
#           v3 therefore adds:
#             exclusion UNK_TEXT : drop rows whose Sequence cell contains
#                                  'unknown' (case-insensitive) anywhere
#             score     RAWLEN   : len(raw cell) INCLUDING the FASTA header
#                                  (the documented header-in-length artifact)
#           and produces the CANONICAL manuscript block (bootstrap SEs,
#           interaction Delta/MDD) for the adjudicated variant.
# =============================================================================

import csv
import hashlib
import os
import sys
import time

import numpy as np

SCRIPT_VERSION = "v3.0-2026-08-08-claude-p1-sec41-final"
CANDIDATE_DIRS = [
    r"C:/Project/AI/EquiPhase",
    r"C:/Project/AI/EquiPhase/data",
    r"C:/Project/EquiPhase",
    r"C:/Project/EquiPhase/data",
]
MATCH_TOL = 0.0025
BOOT_ITERS = 2000
BOOT_SEED = 424242
SEP = "=" * 88

TARGETS = {
    "train": {"LEDGER": (0.5898, 0.6173), "BLOCKED": (0.6291, 0.5756),
              "V1": (0.6306, 0.5634)},
    "val": {"LEDGER": (0.5976, 0.6225), "V1": (0.5948, 0.6076)},
}
EXPECT_CENSUS = {"train": (2539, 1734, 805), "val": (656, 441, 215)}
LEDGER_SARS2 = 0.6406


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_file(name):
    for d in CANDIDATE_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def auroc_mw(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    s_sorted = scores[order]
    i, r = 0, 1.0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (r + (r + (j - i))) / 2.0
        r += (j - i) + 1
        i = j + 1
    n1 = int(labels.sum())
    n2 = len(labels) - n1
    if n1 == 0 or n2 == 0:
        return float("nan"), n1, n2
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2.0)
                 / (n1 * n2)), n1, n2


def hm_se(a, n1, n2):
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    return float(np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a)
                          + (n2 - 1) * (q2 - a * a)) / (n1 * n2)))


def boot_se(scores, labels, rng):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    vals = []
    for _ in range(BOOT_ITERS):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        a, _, _ = auroc_mw(scores[idx], labels[idx])
        vals.append(a)
    vals = np.asarray(vals)
    return float(vals.std(ddof=1)), float(np.percentile(vals, 2.5)), \
        float(np.percentile(vals, 97.5))


def parse_seq_cell(cell):
    header_lines, seq_lines = [], []
    for line in cell.strip().splitlines():
        (header_lines if line.lstrip().startswith(">") else seq_lines).append(line)
    return " ".join(header_lines).strip(), "".join(x.strip() for x in seq_lines)


def load_rows(path, split):
    print(f"\n[LOAD] {split}: {path}")
    print(f"  sha256 = {sha256_file(path)} | size = {os.path.getsize(path)}")
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    header, data = rows[0], rows[1:]
    idx = {name: j for j, name in enumerate(header)}
    col_seq, col_lab = idx["Sequence"], idx["label"]
    recs = []
    for r in data:
        if col_seq >= len(r) or col_lab >= len(r):
            continue
        raw = r[col_seq]
        try:
            y = int(float(r[col_lab]))
        except ValueError:
            continue
        norm = raw.replace("\xa0", " ")
        _, seq = parse_seq_cell(norm)
        recs.append({
            "y": y,
            "complen": len(seq),
            "rawlen": len(raw),
            "unk_bare": raw.strip().upper() == "UNKNOWN",
            "unk_text": "unknown" in raw.lower(),
            "human_resolved": "OS=Homo sapiens" in norm,
            "human_naive": "OS=Homo sapiens" in raw,
            "sars2": ("coronavirus 2" in raw) or ("SARS-CoV-2" in raw)
                     or ("OX=2697049" in raw),
        })
    print(f"  raw rows = {len(recs)} | bare-UNKNOWN = "
          f"{sum(r['unk_bare'] for r in recs)} | contains 'unknown' text = "
          f"{sum(r['unk_text'] for r in recs)}")
    return recs


def strata(recs, mem_key, sc_key):
    hs = [(r[sc_key], r["y"]) for r in recs if r[mem_key]]
    ns = [(r[sc_key], r["y"]) for r in recs if not r[mem_key]]
    a_h, _, _ = auroc_mw([t[0] for t in hs], [t[1] for t in hs]) \
        if hs else (float("nan"), 0, 0)
    a_n, _, _ = auroc_mw([t[0] for t in ns], [t[1] for t in ns]) \
        if ns else (float("nan"), 0, 0)
    return a_h, len(hs), a_n, len(ns)


def nearest(split, a_h, a_n):
    out = []
    for name, (t_h, t_n) in TARGETS[split].items():
        if abs(a_h - t_h) < MATCH_TOL and abs(a_n - t_n) < MATCH_TOL:
            out.append(name)
    return "+".join(out) if out else "NONE"


def run_split(path, split, rng):
    recs = load_rows(path, split)
    excl = {
        "FULL": [r for r in recs if not r["unk_bare"]],
        "UNK_TEXT": [r for r in recs
                     if not (r["unk_bare"] or r["unk_text"])],
    }
    print(f"\n[CENSUS] {split} (expected valid/pos/neg = "
          f"{EXPECT_CENSUS[split]})")
    for tag, rr in excl.items():
        n_pos = sum(r["y"] for r in rr)
        exp = EXPECT_CENSUS[split]
        mark = "<< MATCHES LEDGER CENSUS" if \
            (len(rr), n_pos, len(rr) - n_pos) == exp else ""
        print(f"  {tag:<9}: n={len(rr)} pos={n_pos} neg={len(rr) - n_pos} "
              f"| human_res={sum(r['human_resolved'] for r in rr)} "
              f"| human_naive={sum(r['human_naive'] for r in rr)} "
              f"| sars2={sum(r['sars2'] for r in rr)} {mark}")

    print(f"\n[MATRIX] {split} (A_h | A_nh | Delta | match, tol {MATCH_TOL})")
    print(f"  {'excl':<9} {'species':<9} {'score':<8} "
          f"{'n_h':>5} {'n_nh':>5}  {'A_h':>7} {'A_nh':>7} {'Delta':>8}  match")
    for e_tag, rr in excl.items():
        for m_tag, m_key in (("RESOLVED", "human_resolved"),
                             ("NAIVE", "human_naive")):
            for s_tag, s_key in (("COMPLEN", "complen"), ("RAWLEN", "rawlen")):
                a_h, n_h, a_n, n_nh = strata(rr, m_key, s_key)
                if np.isnan(a_h) or np.isnan(a_n):
                    print(f"  {e_tag:<9} {m_tag:<9} {s_tag:<8} "
                          f"{n_h:>5} {n_nh:>5}  (empty stratum)")
                    continue
                print(f"  {e_tag:<9} {m_tag:<9} {s_tag:<8} {n_h:>5} {n_nh:>5}"
                      f"  {a_h:7.4f} {a_n:7.4f} {a_n - a_h:+8.4f}  "
                      f"{nearest(split, a_h, a_n)}")
        for s_tag, s_key in (("COMPLEN", "complen"), ("RAWLEN", "rawlen")):
            sub = excl[e_tag]
            a, n1, n2 = auroc_mw([r[s_key] for r in sub],
                                 [r["y"] for r in sub])
            print(f"  [{split}/all {e_tag} {s_tag}] n={len(sub)} "
                  f"(pos {n1}/neg {n2}) AUROC = {a:.4f}")
        if split == "train":
            for s_tag, s_key in (("COMPLEN", "complen"), ("RAWLEN", "rawlen")):
                sub = [r for r in excl[e_tag] if r["sars2"]]
                a, n1, n2 = auroc_mw([r[s_key] for r in sub],
                                     [r["y"] for r in sub])
                print(f"  [{split}/sars2 {e_tag} {s_tag}] n={len(sub)} "
                      f"(pos {n1}/neg {n2}) AUROC = {a:.4f} "
                      f"(ledger {LEDGER_SARS2})")

    # canonical manuscript block: UNK_TEXT + RESOLVED + COMPLEN
    print(f"\n[CANONICAL] {split}: UNK_TEXT + RESOLVED + COMPLEN "
          f"(bootstrap {BOOT_ITERS}, seed {BOOT_SEED})")
    rr = excl["UNK_TEXT"]
    out = {}
    groups = [("all", [True] * len(rr)),
              ("human", [r["human_resolved"] for r in rr]),
              ("nonhuman", [not r["human_resolved"] for r in rr])]
    if split == "train":
        groups.append(("sars2", [r["sars2"] for r in rr]))
    for gname, mask in groups:
        sub = [r for r, m in zip(rr, mask) if m]
        sc = np.array([r["complen"] for r in sub], dtype=np.float64)
        yy = np.array([r["y"] for r in sub], dtype=np.int64)
        a, n1, n2 = auroc_mw(sc, yy)
        se_b, lo, hi = boot_se(sc, yy, rng)
        print(f"  {gname:<9} n={len(sub)} (pos {n1}/neg {n2}) "
              f"AUROC={a:.4f} HM_SE={hm_se(a, n1, n2):.4f} "
              f"boot_SE={se_b:.4f} CI95=[{lo:.4f},{hi:.4f}]")
        out[gname] = (a, se_b)
    if "human" in out and "nonhuman" in out:
        (a_h, s_h), (a_n, s_n) = out["human"], out["nonhuman"]
        d = a_n - a_h
        se_d = float(np.sqrt(s_h ** 2 + s_n ** 2))
        print(f"  interaction: Delta={d:+.4f} SE_diff={se_d:.4f} "
              f"MDD={1.96 * se_d:.4f} significant="
              f"{'YES' if abs(d) > 1.96 * se_d else 'NO'}")


def main():
    t0 = time.time()
    print(SEP)
    print("=== CLAUDE PAPER 1 SEC 4.1 FINAL ADJUDICATION (v3) ===")
    print(SEP)
    print(f"  script_version = {SCRIPT_VERSION}")
    print(f"  script_sha256  = {sha256_file(os.path.abspath(__file__))}")
    print(f"  numpy = {np.__version__} | python = {sys.version.split()[0]}")
    rng = np.random.default_rng(BOOT_SEED)
    for split, fname in (("train", "train.tsv"), ("val", "val.tsv")):
        p = find_file(fname)
        if p is None:
            print(f"  ERROR: {fname} not found. ABORT.")
            sys.exit(2)
        run_split(p, split, rng)
    print(f"\n[END] TOTAL WALL-CLOCK = {time.time() - t0:.1f} s")
    print(SEP)
    print("=== END OF CLAUDE PAPER1 SEC4.1 V3 STDOUT ===")
    print(SEP)


if __name__ == "__main__":
    main()
