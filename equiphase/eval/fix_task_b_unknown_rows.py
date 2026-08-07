import os
import sys
import re
import pandas as pd
import numpy as np

sys.path.append(r"C:\Project\AI\EquiPhase")
from equiphase.eval.audit_protocol import compute_auroc

print("=" * 100)
print("=== TASK 1: UNKNOWN SEQUENCE ROW AUDIT & N=656 CLEAN CONF_SEQLEN RE-CALCULATION ===")
print("=" * 100)

val_tsv = r"C:\Project\AI\EquiPhase\equiphase\data\val.tsv"
df_val = pd.read_csv(val_tsv, sep="\t")

print(f"Total rows in val.tsv: {len(df_val)}")

# Identify rows where Sequence contains 'UNKNOWN' or missing indicators
is_unknown = []
clean_aa_lens = []
header_lens = []

for idx, row in df_val.iterrows():
    raw_str = str(row['Sequence'])
    raw_upper = raw_str.upper()
    
    if "UNKNOWN" in raw_upper or "N/A" in raw_upper or len(raw_str.strip()) == 0:
        is_unknown.append(True)
    else:
        is_unknown.append(False)
        
    lines = raw_str.strip().split('\n')
    seq_lines = [l for l in lines if not l.strip().startswith('>')]
    clean_aa = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', "".join(seq_lines).upper())
    clean_aa_lens.append(len(clean_aa))
    
    # Header length
    hdr_lines = [l for l in lines if l.strip().startswith('>')]
    header_lens.append(len("".join(hdr_lines)))

df_val['is_unknown'] = is_unknown
df_val['clean_aa_len'] = clean_aa_lens
df_val['header_len'] = header_lens
y_true = df_val['label'].values

n_unknown = sum(is_unknown)
print(f"\n1. Rows identified as 'UNKNOWN' / Missing Sequence: {n_unknown} / {len(df_val)}")

# Label distribution of UNKNOWN rows
y_unknown = y_true[df_val['is_unknown']]
pos_unk = sum(y_unknown == 1)
neg_unk = sum(y_unknown == 0)
print(f"   Label distribution in 41 UNKNOWN rows: Positive (1) = {pos_unk} | Negative (0) = {neg_unk}")

# Label distribution in remaining valid rows
df_valid = df_val[~df_val['is_unknown']].copy()
y_valid = df_valid['label'].values
print(f"\n2. Remaining Valid Sequence Rows: n = {len(df_valid)}")
print(f"   Label distribution in 656 Valid rows: Positive (1) = {sum(y_valid == 1)} | Negative (0) = {sum(y_valid == 0)}")

def compute_ci(y_t, y_s):
    auc_val = compute_auroc(y_t, y_s)
    n1 = np.sum(y_t == 1)
    n2 = np.sum(y_t == 0)
    q1 = auc_val / (2 - auc_val)
    q2 = 2 * (auc_val**2) / (1 + auc_val)
    se = np.sqrt((auc_val * (1 - auc_val) + (n1 - 1)*(q1 - auc_val**2) + (n2 - 1)*(q2 - auc_val**2)) / (n1 * n2))
    ci_l = max(0.0, auc_val - 1.96 * se)
    ci_h = min(1.0, auc_val + 1.96 * se)
    return auc_val, ci_l, ci_h, (ci_l <= 0.5 <= ci_h)

# (a) True Pure AA CONF_seqlen on n=656 Valid Rows
auc_seq_656, ci_l_656, ci_h_656, inc05_656 = compute_ci(y_valid, df_valid['clean_aa_len'].values)

# (b) Sensitivity on >= 30 AA valid rows
mask_ge30_valid = df_valid['clean_aa_len'] >= 30
auc_ge30_v, ci_l_ge30_v, ci_h_ge30_v, inc05_ge30_v = compute_ci(y_valid[mask_ge30_valid], df_valid.loc[mask_ge30_valid, 'clean_aa_len'].values)

# (c) CONF_header on n=656 Valid Rows
auc_hdr_656, ci_l_hdr_656, ci_h_hdr_656, inc05_hdr_656 = compute_ci(y_valid, df_valid['header_len'].values)

# (d) Missingness Indicator AUROC (is_unknown predicting label)
auc_missing, ci_l_miss, ci_h_miss, inc05_miss = compute_ci(y_true, df_val['is_unknown'].astype(float).values)

print("\n" + "=" * 80)
print("=== FINAL EMPIRICAL RESULTS AFTER EXCLUDING 41 UNKNOWN ROWS ===")
print("=" * 80)
print(f"1. TRUE PURE AA SEQUENCE CONF_seqlen (n=656 Valid Rows):")
print(f"   AUROC = {auc_seq_656:.4f} [95% CI: {ci_l_656:.4f}, {ci_h_656:.4f}] | Includes 0.5: {inc05_656}")

print(f"\n2. TRUE PURE AA SEQUENCE CONF_seqlen (>= 30 AA Valid Rows, n={mask_ge30_valid.sum()}):")
print(f"   AUROC = {auc_ge30_v:.4f} [95% CI: {ci_l_ge30_v:.4f}, {ci_h_ge30_v:.4f}] | Includes 0.5: {inc05_ge30_v}")

print(f"\n3. HEADER / ANNOTATION LENGTH CONF_header (n=656 Valid Rows):")
print(f"   AUROC = {auc_hdr_656:.4f} [95% CI: {ci_l_hdr_656:.4f}, {ci_h_hdr_656:.4f}] | Includes 0.5: {inc05_hdr_656}")

print(f"\n4. MISSINGNESS INDICATOR CONF_missing (n=697 Total Rows):")
print(f"   AUROC = {auc_missing:.4f} [95% CI: {ci_l_miss:.4f}, {ci_h_miss:.4f}] | Includes 0.5: {inc05_miss}")
