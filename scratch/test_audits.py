import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge

sys.path.append("D:/AI/EquiPhase")
import upaf

# Load datasets
df_train = pd.read_csv("equiphase/data/train.tsv", sep="\t")
df_val = pd.read_csv("equiphase/data/val.tsv", sep="\t")
df_test = pd.read_csv("equiphase/data/test.tsv", sep="\t")
df_llps = pd.concat([df_train, df_val, df_test], ignore_index=True)

with open("equiphase/data/esm2_embeddings.pkl", "rb") as f:
    esm_embeddings = pickle.load(f)
    
X_esm = np.array([esm_embeddings[seq] for seq in df_llps['Sequence']])

# Task B setup: Include cluster_id as a feature to simulate memorization leakage
c_sat = df_llps['parsed_c_sat'].values
salt = df_llps['parsed_salt'].values
ph = df_llps['parsed_ph'].values
temp = df_llps['parsed_temp'].values
cluster_id = df_llps['cluster_id'].values

c_sat_scaled = (c_sat - c_sat.mean()) / (c_sat.std() + 1e-5)
salt_scaled = (salt - salt.mean()) / (salt.std() + 1e-5)
ph_scaled = (ph - ph.mean()) / (ph.std() + 1e-5)
temp_scaled = (temp - temp.mean()) / (temp.std() + 1e-5)
cluster_scaled = (cluster_id - cluster_id.mean()) / (cluster_id.std() + 1e-5)

X_b = np.column_stack([X_esm, c_sat_scaled, salt_scaled, ph_scaled, temp_scaled, cluster_scaled])
y_b = df_llps['label'].values
groups_b = df_llps['cluster_id'].values

print("--- Testing Task B (RandomForestClassifier, depth=12) ---")
res_b = upaf.audit(
    model_class=RandomForestClassifier,
    model_args={"n_estimators": 50, "max_depth": 12, "random_state": 42},
    X=X_b,
    y=y_b,
    groups=groups_b,
    confounds=c_sat_scaled,
    target_features=None,
    n_seeds=3,
    task_name="test_b_rf"
)

# Task C setup: Use amino acid counts to predict length (clean linear relationship)
def get_aa_counts(seq):
    # Extract only the amino acids part
    amino_acids = "".join(seq.split('\n')[1:])
    counts = []
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        counts.append(amino_acids.count(aa))
    return counts

X_c = np.array([get_aa_counts(seq) for seq in df_llps['Sequence']], dtype=float)
seq_lens = []
for seq in df_llps['Sequence']:
    parts = seq.split('\n')
    amino_acids = "".join(parts[1:])
    seq_lens.append(len(amino_acids))
y_c = np.array(seq_lens, dtype=float)

print("--- Testing Task C (Ridge, AA counts) ---")
res_c = upaf.audit(
    model_class=Ridge,
    model_args={"alpha": 1.0},
    X=X_c,
    y=y_c,
    groups=groups_b,
    confounds=None,
    target_features=None,
    n_seeds=3,
    task_name="test_c_ridge"
)
