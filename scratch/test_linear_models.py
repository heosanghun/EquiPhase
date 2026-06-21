import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import KFold

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

# Task B setup
c_sat = df_llps['parsed_c_sat'].values
salt = df_llps['parsed_salt'].values
ph = df_llps['parsed_ph'].values
temp = df_llps['parsed_temp'].values

c_sat_scaled = (c_sat - c_sat.mean()) / (c_sat.std() + 1e-5)
salt_scaled = (salt - salt.mean()) / (salt.std() + 1e-5)
ph_scaled = (ph - ph.mean()) / (ph.std() + 1e-5)
temp_scaled = (temp - temp.mean()) / (temp.std() + 1e-5)

X_conds = np.column_stack([c_sat_scaled, salt_scaled, ph_scaled, temp_scaled])
X_b = np.hstack([X_esm, X_conds])
y_b = df_llps['label'].values
groups_b = df_llps['cluster_id'].values

# Task C setup
seq_lens = []
for seq in df_llps['Sequence']:
    parts = seq.split('\n')
    amino_acids = "".join(parts[1:])
    seq_lens.append(len(amino_acids))
y_c = np.array(seq_lens, dtype=float)

print("--- Testing Task B (LogisticRegression) ---")
res_b = upaf.audit(
    model_class=LogisticRegression,
    model_args={"max_iter": 1000, "random_state": 42},
    X=X_b,
    y=y_b,
    groups=groups_b,
    confounds=c_sat_scaled,
    target_features=None,
    n_seeds=3,
    task_name="test_b_temp"
)

print("--- Testing Task C (Ridge) ---")
res_c = upaf.audit(
    model_class=Ridge,
    model_args={"alpha": 1.0},
    X=X_esm,
    y=y_c,
    groups=groups_b,
    confounds=None,
    target_features=None,
    n_seeds=3,
    task_name="test_c_temp"
)
