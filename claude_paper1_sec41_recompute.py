# =============================================================================
# claude_paper1_sec41_recompute.py
# Author  : Claude (external auditor, System 2)  --  SEALED SCRIPT, DO NOT MODIFY
# Version : v1.0 (2026-08-08)
# Purpose : Independent recomputation of Paper 1 (UPAF) Section 4.1 Task B
#           species-stratified sequence-length AUROCs, to adjudicate the
#           ledger values whose only surviving source is the conversation
#           record (original task-1262.log was tampered):
#               train Homo sapiens n=1302 : 0.5898
#               train Non-Human    n=1237 : 0.6173
#               train SARS-CoV-2   n= 253 : 0.6406
#               val   Homo sapiens n= 483 : 0.5976
#               val   Non-Human    n= 173 : 0.6225
#
# DESIGN
#   PHASE 0  schema diagnostics (column names, dtypes, first rows) printed
#            verbatim so parsing decisions are auditable.
#   PHASE 1  row-validity + species parsing with the established rules, each
#            count checked against the sealed expected census. Mismatches are
#            PRINTED, never silently absorbed.
#   PHASE 2  AUROC (exact Mann-Whitney with tie correction) per stratum,
#            Hanley-McNeil SE + stratified bootstrap SE (2000 iters, seeded),
#            interaction Delta / MDD, side-by-side vs ledger.
#
# PARSING RULES (frozen; taken from the audited pipeline record)
#   - Normalize non-breaking spaces \xa0 -> ' ' everywhere before matching.
#   - A row is INVALID if its sequence cell (stripped, upper) == "UNKNOWN".
#   - If the sequence cell embeds a FASTA header, the header is every line
#     starting with '>'; the AA sequence is the remaining lines concatenated.
#   - seqlen = number of characters in the pure AA sequence.
#   - species: human iff the FASTA/OS field contains 'OS=Homo sapiens'
#     (post-normalization); SARS-CoV-2 iff it contains 'OS=Severe acute
#     respiratory syndrome coronavirus 2' or 'SARS-CoV-2' or 'OX=2697049'.
#   - label column: auto-detected as the column whose non-null values are a
#     subset of {0,1} (or {'0','1'}); positives are label == 1.
# =============================================================================

import csv
import hashlib
import os
import sys
import time

import numpy as np

SCRIPT_VERSION = "v1.0-2026-08-08-claude-p1-sec41"
CANDIDATE_DIRS = [
    r"C:/Project/AI/EquiPhase",
    r"C:/Project/AI/EquiPhase/data",
    r"C:/Project/EquiPhase",
    r"C:/Project/EquiPhase/data",
]
BOOT_ITERS = 2000
BOOT_SEED = 424242

LEDGER = {
    ("train", "human"): (1302, 0.5898),
    ("train", "nonhuman"): (1237, 0.6173),
    ("train", "sars2"): (253, 0.6406),
    ("val", "human"): (483, 0.5976),
    ("val", "nonhuman"): (173, 0.6225),
    ("val", "all"): (656, 0.6017),
    ("train", "all"): (2539, 0.6010),
}
CENSUS = {
    "val": {"raw": 697, "valid": 656, "pos": 441, "neg": 215},
    "train": {"raw": 2554, "valid": 2539, "pos": 1734, "neg": 805},
}
SEP = "=" * 88


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
    """Exact Mann-Whitney AUROC with midrank tie handling."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    s_sorted = scores[order]
    i = 0
    r = 1.0
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
    auc = (ranks[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n2)
    return float(auc), n1, n2


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
        ip = rng.choice(pos, size=len(pos), replace=True)
        ineg = rng.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([ip, ineg])
        a, _, _ = auroc_mw(scores[idx], labels[idx])
        vals.append(a)
    vals = np.asarray(vals)
    return float(vals.std(ddof=1)), float(np.percentile(vals, 2.5)), \
        float(np.percentile(vals, 97.5))


def normalize(s):
    return s.replace("\xa0", " ") if isinstance(s, str) else s


def parse_seq_cell(cell):
    """Return (header_text, pure_sequence). Header = lines starting with '>'."""
    cell = cell.strip()
    header_lines, seq_lines = [], []
    for line in cell.splitlines():
        (header_lines if line.lstrip().startswith(">") else seq_lines).append(line)
    return " ".join(header_lines).strip(), "".join(
        x.strip() for x in seq_lines)


def load_split(path, split_name):
    print(f"\n[PHASE 0] SCHEMA DIAGNOSTICS -- {split_name} : {path}")
    print(f"  sha256 = {sha256_file(path)} | size = {os.path.getsize(path)}")
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)
    header, data = rows[0], rows[1:]
    print(f"  columns ({len(header)}): {header}")
    print(f"  raw data rows = {len(data)} "
          f"(expected {CENSUS[split_name]['raw']})")
    for k in range(min(2, len(data))):
        prev = [c[:60].replace("\n", "\\n") for c in data[k]]
        print(f"  row {k}: {prev}")

    # column auto-detection
    ncol = len(header)
    label_col = seq_col = None
    for j in range(ncol):
        vals = {r[j].strip() for r in data if j < len(r) and r[j].strip()}
        if vals and vals <= {"0", "1", "0.0", "1.0"}:
            label_col = j
    lens = [np.mean([len(r[j]) for r in data[:200] if j < len(r)])
            for j in range(ncol)]
    seq_col = int(np.argmax(lens))
    if seq_col == label_col:
        print("  ERROR: sequence/label column collision. ABORT with schema "
              "above; auditor will issue v2 with explicit columns.")
        sys.exit(2)
    print(f"  auto-detected: label_col = {label_col} "
          f"({header[label_col] if label_col is not None else 'NONE'}), "
          f"seq_col = {seq_col} ({header[seq_col]})")
    if label_col is None:
        print("  ERROR: no binary label column found. ABORT.")
        sys.exit(2)

    # species source: prefer a dedicated column containing 'OS=' or species
    # names; otherwise use the header embedded in the sequence cell.
    species_col = None
    for j in range(ncol):
        if j in (seq_col, label_col):
            continue
        sample = " ".join(normalize(r[j]) for r in data[:300] if j < len(r))
        if "OS=" in sample or "Homo sapiens" in sample:
            species_col = j
            break
    print(f"  species source: "
          f"{'column ' + str(species_col) + ' (' + header[species_col] + ')' if species_col is not None else 'FASTA header inside seq cell'}")

    recs = []
    n_unknown = 0
    for r in data:
        if seq_col >= len(r) or label_col >= len(r):
            continue
        raw_seq = normalize(r[seq_col])
        if raw_seq.strip().upper() == "UNKNOWN":
            n_unknown += 1
            continue
        head_embed, seq = parse_seq_cell(raw_seq)
        sp_text = normalize(r[species_col]) if species_col is not None \
            else head_embed
        sp_text = sp_text or head_embed
        try:
            y = int(float(r[label_col]))
        except ValueError:
            continue
        is_human = "OS=Homo sapiens" in sp_text or \
            (species_col is not None and sp_text.strip() == "Homo sapiens")
        is_sars2 = ("coronavirus 2" in sp_text) or ("SARS-CoV-2" in sp_text) \
            or ("OX=2697049" in sp_text)
        recs.append((len(seq), y, is_human, is_sars2))

    print(f"\n[PHASE 1] CENSUS -- {split_name}")
    exp = CENSUS[split_name]
    n_pos = sum(1 for t in recs if t[1] == 1)
    n_neg = len(recs) - n_pos
    n_h = sum(1 for t in recs if t[2])
    n_s = sum(1 for t in recs if t[3])
    for name, got, want in [("UNKNOWN excluded", n_unknown,
                             exp["raw"] - exp["valid"]),
                            ("valid rows", len(recs), exp["valid"]),
                            ("positives", n_pos, exp["pos"]),
                            ("negatives", n_neg, exp["neg"])]:
        print(f"  {name}: {got} (expected {want}) "
              f"{'OK' if got == want else '** MISMATCH **'}")
    print(f"  human = {n_h} | non-human = {len(recs) - n_h} "
          f"| SARS-CoV-2 = {n_s}")
    return recs


def stratum_report(recs, mask, split, key, rng):
    sub = [t for t, m in zip(recs, mask) if m]
    scores = np.array([t[0] for t in sub], dtype=np.float64)
    labels = np.array([t[1] for t in sub], dtype=np.int64)
    a, n1, n2 = auroc_mw(scores, labels)
    se_h = hm_se(a, n1, n2) if not np.isnan(a) else float("nan")
    se_b, lo, hi = boot_se(scores, labels, rng)
    exp_n, exp_a = LEDGER.get((split, key), (None, None))
    flag = ""
    if exp_a is not None:
        flag = "OK" if abs(a - exp_a) < 5e-4 else \
            f"** DIFFERS from ledger {exp_a:.4f} by {a - exp_a:+.4f} **"
    print(f"  [{split}/{key}] n={len(sub)} (pos {n1}/neg {n2}) | "
          f"AUROC = {a:.4f} | HM SE = {se_h:.4f} | boot SE = {se_b:.4f} "
          f"| boot 95% CI [{lo:.4f}, {hi:.4f}] | ledger n={exp_n} -> {flag}")
    return a, se_b


def main():
    t0 = time.time()
    print(SEP)
    print("=== CLAUDE PAPER 1 SEC 4.1 INDEPENDENT RECOMPUTATION (Task B strata) ===")
    print(SEP)
    print(f"  script_version = {SCRIPT_VERSION}")
    print(f"  script_sha256  = {sha256_file(os.path.abspath(__file__))}")
    print(f"  numpy = {np.__version__} | python = {sys.version.split()[0]}")

    paths = {}
    for split, fname in [("train", "train.tsv"), ("val", "val.tsv")]:
        p = find_file(fname)
        if p is None:
            print(f"  ERROR: {fname} not found in {CANDIDATE_DIRS}. "
                  f"Place the file or report its true path to the auditor.")
            sys.exit(2)
        paths[split] = p

    rng = np.random.default_rng(BOOT_SEED)
    results = {}
    for split in ("train", "val"):
        recs = load_split(paths[split], split)
        print(f"\n[PHASE 2] STRATIFIED AUROC -- {split} "
              f"(bootstrap {BOOT_ITERS} iters, seed {BOOT_SEED})")
        results[(split, "all")] = stratum_report(
            recs, [True] * len(recs), split, "all", rng)
        results[(split, "human")] = stratum_report(
            recs, [t[2] for t in recs], split, "human", rng)
        results[(split, "nonhuman")] = stratum_report(
            recs, [not t[2] for t in recs], split, "nonhuman", rng)
        if split == "train":
            results[(split, "sars2")] = stratum_report(
                recs, [t[3] for t in recs], split, "sars2", rng)

    print(f"\n[PHASE 2b] INTERACTION TEST (train, human vs non-human)")
    a_h, se_h = results[("train", "human")]
    a_n, se_n = results[("train", "nonhuman")]
    delta = a_n - a_h
    se_d = float(np.sqrt(se_h ** 2 + se_n ** 2))
    mdd = 1.96 * se_d
    print(f"  Delta = {a_n:.4f} - {a_h:.4f} = {delta:+.4f} | "
          f"SE_diff = {se_d:.4f} | MDD = {mdd:.4f} | "
          f"significant: {'YES' if abs(delta) > mdd else 'NO'} "
          f"(ledger: Delta=+0.0275, MDD=0.0465, not significant)")

    print(f"\n[END] TOTAL WALL-CLOCK = {time.time() - t0:.1f} s")
    print(SEP)
    print("=== END OF CLAUDE PAPER1 SEC4.1 STDOUT ===")
    print(SEP)


if __name__ == "__main__":
    main()
