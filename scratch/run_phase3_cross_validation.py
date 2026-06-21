import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge

# Ensure workspace is in path
sys.path.append("D:/AI/EquiPhase")

import upaf
from iss_data import parse_pdb

def run_task_a():
    print("\n========================================================")
    print("TASK A: FOLD-SWITCH STABILITY MARGIN PREDICTION (LEAKY)")
    print("========================================================\n")
    
    # Load dataset
    df_pairs = pd.read_csv("data/benchmark_pairs.csv")
    df_plddt = pd.read_csv("data/imputed_benchmark_plddt.csv")
    
    # Parse PDB files to extract sequences for clustering
    unique_pdbs = set(df_pairs['pdb1'].tolist() + df_pairs['pdb2'].tolist())
    parsed_seqs = {}
    for pdb in unique_pdbs:
        path = f"data/clean_chains/{pdb}.pdb"
        seq, coords = parse_pdb(path)
        if seq is None:
            seq = "M" * 100
        parsed_seqs[pdb] = seq
        
    # Sequence Clustering Proxy (30% global identity)
    chain_list = list(unique_pdbs)
    chain_to_fam = {}
    fam_counter = 0
    visited = set()
    for chain1 in chain_list:
        if chain1 in visited:
            continue
        current_fam = f"fam_{fam_counter}"
        fam_counter += 1
        queue = [chain1]
        visited.add(chain1)
        while queue:
            curr = queue.pop(0)
            chain_to_fam[curr] = current_fam
            seq_curr = parsed_seqs[curr]
            
            for other in chain_list:
                if other in visited:
                    continue
                seq_other = parsed_seqs[other]
                len_ratio = len(seq_curr) / len(seq_other)
                if len_ratio < 0.5 or len_ratio > 2.0:
                    continue
                n1, n2 = len(seq_curr), len(seq_other)
                min_len = min(n1, n2)
                mismatches = sum(1 for a, b in zip(seq_curr[:min_len], seq_other[:min_len]) if a != b)
                identity = (min_len - mismatches) / max(n1, n2)
                if identity >= 0.30:
                    visited.add(other)
                    queue.append(other)
                    
    df_pairs['family_id'] = [chain_to_fam[row['pdb1']] for _, row in df_pairs.iterrows()]
    groups = df_pairs['family_id'].values
    y = df_pairs['is_switcher'].values
    
    # Features: Column 0: pair_rmsd (target-dependent leak), 1: plddt1, 2: plddt2
    plddt1 = df_plddt['plddt1'].fillna(df_plddt['plddt1'].mean()).values
    plddt2 = df_plddt['plddt2'].fillna(df_plddt['plddt2'].mean()).values
    X = np.column_stack([df_pairs['pair_rmsd'].values, plddt1, plddt2])
    
    # Run UPAF audit
    audit_res = upaf.audit(
        model_class=RandomForestClassifier,
        model_args={"n_estimators": 30, "max_depth": 5, "random_state": 42},
        X=X,
        y=y,
        groups=groups,
        confounds=df_pairs['pair_rmsd'].values,
        target_features=[0], # Column 0 (pair_rmsd) is target-dependent leak
        n_seeds=5,
        task_name="task_a_fold_switch"
    )
    return audit_res

def run_task_b_and_c():
    print("\n========================================================")
    print("LOADING LLPS DATASETS FOR TASK B & C")
    print("========================================================\n")
    
    # Load and merge train, val, test datasets
    df_train = pd.read_csv("equiphase/data/train.tsv", sep="\t")
    df_val = pd.read_csv("equiphase/data/val.tsv", sep="\t")
    df_test = pd.read_csv("equiphase/data/test.tsv", sep="\t")
    df_llps = pd.concat([df_train, df_val, df_test], ignore_index=True)
    
    # Load ESM embeddings
    with open("equiphase/data/esm2_embeddings.pkl", "rb") as f:
        esm_embeddings = pickle.load(f)
        
    X_esm = np.array([esm_embeddings[seq] for seq in df_llps['Sequence']])
    groups_b = df_llps['cluster_id'].values
    
    print("TASK B: LLPS BINARY STATUS PREDICTION (OOD SENSITIVE)")
    print("--------------------------------------------------------\n")
    
    # Compute cluster-mean ESM embeddings to simulate cluster memorization leakage
    cluster_means = {cid: X_esm[groups_b == cid].mean(axis=0) for cid in np.unique(groups_b)}
    X_cm = np.array([cluster_means[cid] for cid in groups_b])
    
    # Extract and normalize conditions
    c_sat = df_llps['parsed_c_sat'].values
    salt = df_llps['parsed_salt'].values
    ph = df_llps['parsed_ph'].values
    temp = df_llps['parsed_temp'].values
    
    c_sat_scaled = (c_sat - c_sat.mean()) / (c_sat.std() + 1e-5)
    salt_scaled = (salt - salt.mean()) / (salt.std() + 1e-5)
    ph_scaled = (ph - ph.mean()) / (ph.std() + 1e-5)
    temp_scaled = (temp - temp.mean()) / (temp.std() + 1e-5)
    
    X_conds = np.column_stack([c_sat_scaled, salt_scaled, ph_scaled, temp_scaled])
    # Combine cluster-mean ESM embeddings with conditions
    X_b = np.hstack([X_cm, X_conds])
    y_b = df_llps['label'].values
    
    audit_res_b = upaf.audit(
        model_class=RandomForestClassifier,
        model_args={"n_estimators": 50, "max_depth": None, "random_state": 42},
        X=X_b,
        y=y_b,
        groups=groups_b,
        confounds=c_sat_scaled, # Control for concentration
        target_features=None,
        n_seeds=5,
        task_name="task_b_llps_leak"
    )
    
    print("TASK C: SEQUENCE LENGTH PREDICTION (CLEAN GENERALIZATION)")
    print("--------------------------------------------------------\n")
    
    # Extract sequence lengths
    seq_lens = []
    for seq in df_llps['Sequence']:
        parts = seq.split('\n')
        amino_acids = "".join(parts[1:])
        seq_lens.append(len(amino_acids))
        
    y_c = np.array(seq_lens, dtype=float) # Cast to float to auto-detect regression
    
    # Compute 20 amino acid counts as the features (clean linear relationship)
    def get_aa_counts(seq):
        amino_acids = "".join(seq.split('\n')[1:])
        counts = []
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            counts.append(amino_acids.count(aa))
        return counts
        
    X_c = np.array([get_aa_counts(seq) for seq in df_llps['Sequence']], dtype=float)
    
    audit_res_c = upaf.audit(
        model_class=Ridge,
        model_args={"alpha": 1.0},
        X=X_c,
        y=y_c,
        groups=groups_b,
        confounds=None,
        target_features=None,
        n_seeds=5,
        task_name="task_c_seq_len_clean"
    )
    
    return audit_res_b, audit_res_c

def main():
    res_a = run_task_a()
    res_b, res_c = run_task_b_and_c()
    
    results = {
        "task_a": res_a,
        "task_b": res_b,
        "task_c": res_c
    }
    
    # Save results
    save_path = "D:/AI/EquiPhase/data/upaf_cross_validation_results.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n========================================================")
    print("CROSS-VALIDATION EVALUATION COMPLETE")
    print("========================================================")
    print(f"Results successfully saved to {save_path}")

if __name__ == "__main__":
    main()
