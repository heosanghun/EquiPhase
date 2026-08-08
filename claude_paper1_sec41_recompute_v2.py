# =============================================================================
# claude_paper1_sec41_recompute_v2.py
# Author  : Claude (external auditor, System 2)  --  SEALED SCRIPT, DO NOT MODIFY
# Version : v2.0 (2026-08-08)
# Purpose : Forensic adjudication of Paper 1 Section 4.1 Task B strata AUROCs.
#           v1 established: (a) val census + val/all AUROC reproduce exactly;
#           (b) train census mismatches (0 "UNKNOWN" cells vs 15 expected);
#           (c) train strata REVERSE direction vs the ledger record.
#           v2 locates WHICH parsing variant generates each historical value
#           by computing the strata AUROCs under a controlled variant matrix.
#
# VARIANT AXES (all combinations reported; nothing hidden)
#   inclusion : FULL      = all rows whose Sequence cell != "UNKNOWN" (v1 rule)
#               DROPSHORT = additionally drop rows whose parsed AA sequence is
#                           shorter than 10 chars (candidate for the missing
#                           15-row exclusion in train)
#   species   : RESOLVED  = normalize \xa0 -> ' ' BEFORE matching 'OS=Homo sapiens'
#               NAIVE     = match on the RAW string (pre-fix behaviour; nbsp
#                           human rows fall into non-human)
#   score     : COMPLEN   = computed pure AA length (headers stripped)
#               COLLEN    = the file's own 'Sequence length' column
#
# HISTORICAL TARGET SETS (train human / non-human)
#   LEDGER  = 0.5898 / 0.6173   (conversation-record values; original log tampered)
#   BLOCKED = 0.6291 / 0.5756   (previously classified as fabricated)
#   V1      = 0.6306 / 0.5634   (v1 sealed recomputation, FULL+RESOLVED+COMPLEN)
#   val targets: LEDGER 0.5976 / 0.6225 ; V1 0.5948 / 0.6076
# A variant "matches" a target set if both strata differ by < 0.0025.
# =============================================================================

import csv
import hashlib
import os
import sys
import time

import numpy as np

SCRIPT_VERSION = "v2.0-2026-08-08-claude-p1-sec41-forensic"
CANDIDATE_DIRS = [
    r"C:/Project/AI/EquiPhase",
    r"C:/Project/AI/EquiPhase/data",
    r"C:/Project/EquiPhase",
    r"C:/Project/EquiPhase/data",
]
SHORT_SEQ_THRESH = 10
MATCH_TOL = 0.0025
SEP = "=" * 88

TARGETS = {
    "train": {"LEDGER": (0.5898, 0.6173), "BLOCKED": (0.6291, 0.5756),
              "V1": (0.6306, 0.5634)},
    "val": {"LEDGER": (0.5976, 0.6225), "V1": (0.5948, 0.6076)},
}


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
    for need in ("Sequence", "label"):
        if need not in idx:
            print(f"  ERROR: column '{need}' missing. ABORT.")
            sys.exit(2)
    col_seq = idx["Sequence"]
    col_lab = idx["label"]
    col_len = idx.get("Sequence length")
    col_psid = idx.get("PSID", 0)
    recs = []
    for r in data:
        if col_seq >= len(r) or col_lab >= len(r):
            continue
        raw = r[col_seq]
        try:
            y = int(float(r[col_lab]))
        except ValueError:
            continue
        head, seq = parse_seq_cell(raw.replace("\xa0", " "))
        head_raw, _ = parse_seq_cell(raw)  # header WITHOUT nbsp normalization
        col_len_val = None
        if col_len is not None and col_len < len(r):
            try:
                col_len_val = float(r[col_len])
            except ValueError:
                col_len_val = None
        recs.append({
            "psid": r[col_psid] if col_psid < len(r) else "",
            "raw": raw,
            "y": y,
            "seq": seq,
            "complen": len(seq),
            "collen": col_len_val,
            "unknown": raw.strip().upper() == "UNKNOWN",
            "human_resolved": "OS=Homo sapiens" in head.replace("\xa0", " ")
                              or "OS=Homo sapiens" in raw.replace("\xa0", " "),
            "human_naive": "OS=Homo sapiens" in head_raw
                           or "OS=Homo sapiens" in raw,
            "sars2": ("coronavirus 2" in raw) or ("SARS-CoV-2" in raw)
                     or ("OX=2697049" in raw),
        })
    print(f"  raw rows = {len(recs)} | UNKNOWN cells = "
          f"{sum(r['unknown'] for r in recs)}")
    return recs


def short_seq_diagnostics(recs, split):
    print(f"\n[DIAG] {split}: rows with parsed AA length < {SHORT_SEQ_THRESH} "
          f"or 'unknown' text (candidates for the missing exclusion set)")
    hits = [r for r in recs if (not r["unknown"]) and
            (r["complen"] < SHORT_SEQ_THRESH
             or "unknown" in r["raw"].lower())]
    print(f"  count = {len(hits)}")
    for r in hits[:25]:
        prev = r["raw"][:70].replace("\n", "\\n").replace("\t", " ")
        print(f"    PSID={r['psid']} y={r['y']} complen={r['complen']} "
              f"collen={r['collen']} human_res={r['human_resolved']} "
              f"cell='{prev}'")
    return hits


def strata_auroc(recs, member_key, score_key):
    """Return (A_human, n_h, A_nonhuman, n_nh) for given membership/score."""
    sub = [r for r in recs if r[score_key] is not None]
    hs = [r[score_key] for r in sub if r[member_key]]
    hy = [r["y"] for r in sub if r[member_key]]
    ns = [r[score_key] for r in sub if not r[member_key]]
    ny = [r["y"] for r in sub if not r[member_key]]
    a_h, _, _ = auroc_mw(hs, hy) if hs else (float("nan"), 0, 0)
    a_n, _, _ = auroc_mw(ns, ny) if ns else (float("nan"), 0, 0)
    return a_h, len(hs), a_n, len(ns)


def nearest_target(split, a_h, a_n):
    best = "NONE"
    for name, (t_h, t_n) in TARGETS[split].items():
        if abs(a_h - t_h) < MATCH_TOL and abs(a_n - t_n) < MATCH_TOL:
            best = name
    return best


def run_split(path, split):
    recs_all = load_rows(path, split)
    short_seq_diagnostics(recs_all, split)

    base = [r for r in recs_all if not r["unknown"]]
    dropshort = [r for r in base if r["complen"] >= SHORT_SEQ_THRESH]
    print(f"\n[CENSUS] {split}: FULL(valid) = {len(base)} | "
          f"DROPSHORT = {len(dropshort)}")
    for tag, rr in (("FULL", base), ("DROPSHORT", dropshort)):
        n_pos = sum(r["y"] for r in rr)
        print(f"  {tag}: pos={n_pos} neg={len(rr) - n_pos} "
              f"| human_resolved={sum(r['human_resolved'] for r in rr)} "
              f"| human_naive={sum(r['human_naive'] for r in rr)} "
              f"| sars2={sum(r['sars2'] for r in rr)}")

    print(f"\n[MATRIX] {split}: strata AUROC under all variants "
          f"(A_human | A_nonhuman | Delta | nearest historical set, "
          f"tol {MATCH_TOL})")
    print(f"  {'inclusion':<10} {'species':<9} {'score':<8} "
          f"{'n_h':>5} {'n_nh':>5}  {'A_h':>7} {'A_nh':>7} {'Delta':>8}  match")
    for inc_tag, rr in (("FULL", base), ("DROPSHORT", dropshort)):
        for mem_tag, mem_key in (("RESOLVED", "human_resolved"),
                                 ("NAIVE", "human_naive")):
            for sc_tag, sc_key in (("COMPLEN", "complen"),
                                   ("COLLEN", "collen")):
                a_h, n_h, a_n, n_nh = strata_auroc(rr, mem_key, sc_key)
                if np.isnan(a_h) or np.isnan(a_n):
                    print(f"  {inc_tag:<10} {mem_tag:<9} {sc_tag:<8} "
                          f"{n_h:>5} {n_nh:>5}  (empty stratum or unusable "
                          f"score column)")
                    continue
                m = nearest_target(split, a_h, a_n)
                print(f"  {inc_tag:<10} {mem_tag:<9} {sc_tag:<8} "
                      f"{n_h:>5} {n_nh:>5}  {a_h:7.4f} {a_n:7.4f} "
                      f"{a_n - a_h:+8.4f}  {m}")

    # overall AUROC per score definition, for anchoring
    for sc_tag, sc_key in (("COMPLEN", "complen"), ("COLLEN", "collen")):
        sub = [r for r in base if r[sc_key] is not None]
        a, n1, n2 = auroc_mw([r[sc_key] for r in sub], [r["y"] for r in sub])
        print(f"  [{split}/all FULL {sc_tag}] n={len(sub)} "
              f"(pos {n1}/neg {n2}) AUROC = {a:.4f}")


def main():
    t0 = time.time()
    print(SEP)
    print("=== CLAUDE PAPER 1 SEC 4.1 FORENSIC VARIANT MATRIX (v2) ===")
    print(SEP)
    print(f"  script_version = {SCRIPT_VERSION}")
    print(f"  script_sha256  = {sha256_file(os.path.abspath(__file__))}")
    print(f"  numpy = {np.__version__} | python = {sys.version.split()[0]}")
    for split, fname in (("train", "train.tsv"), ("val", "val.tsv")):
        p = find_file(fname)
        if p is None:
            print(f"  ERROR: {fname} not found. ABORT.")
            sys.exit(2)
        run_split(p, split)
    print(f"\n[END] TOTAL WALL-CLOCK = {time.time() - t0:.1f} s")
    print(SEP)
    print("=== END OF CLAUDE PAPER1 SEC4.1 V2 STDOUT ===")
    print(SEP)


if __name__ == "__main__":
    main()
