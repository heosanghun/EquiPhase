import pandas as pd
import numpy as np
import os
import sys
from collections import Counter
import time
sys.path.append("D:/AI/EquiPhase")
from iss_data import parse_pdb

def get_sliding_identity(seq1, seq2):
    n1, n2 = len(seq1), len(seq2)
    if n1 == 0 or n2 == 0:
        return 0.0
        
    len_ratio = n1 / n2
    if len_ratio < 0.5 or len_ratio > 2.0:
        return 0.0
        
    # Fast 5-mer filter
    k = 5
    if n1 >= k and n2 >= k:
        kmers1 = set(seq1[i:i+k] for i in range(n1 - k + 1))
        shared = False
        for i in range(n2 - k + 1):
            if seq2[i:i+k] in kmers1:
                shared = True
                break
        if not shared:
            return 0.0
            
    # Fast frequency-based upper bound filter
    c1 = Counter(seq1)
    c2 = Counter(seq2)
    max_possible_matches = sum(min(c1[char], c2[char]) for char in c1)
    if max_possible_matches / max(n1, n2) < 0.30:
        return 0.0
        
    best_identity = 0.0
    for shift in range(-n1 + 1, n2):
        matches = 0
        for i in range(n1):
            j = i + shift
            if 0 <= j < n2:
                if seq1[i] == seq2[j]:
                    matches += 1
        identity = matches / max(n1, n2)
        if identity > best_identity:
            best_identity = identity
    return best_identity

def main():
    df_pairs = pd.read_csv("data/benchmark_pairs.csv")
    unique_pdbs = list(set(df_pairs['pdb1'].tolist() + df_pairs['pdb2'].tolist()))
    
    parsed_seqs = {}
    for pdb in unique_pdbs:
        path = f"data/clean_chains/{pdb}.pdb"
        seq, _ = parse_pdb(path)
        parsed_seqs[pdb] = seq
        
    print("Building sequence identity graph...")
    start_time = time.time()
    adj = {pdb: set() for pdb in unique_pdbs}
    
    # 1. Add edges for sequence identity >= 30%
    for i, p1 in enumerate(unique_pdbs):
        seq1 = parsed_seqs[p1]
        for j in range(i + 1, len(unique_pdbs)):
            p2 = unique_pdbs[j]
            seq2 = parsed_seqs[p2]
            
            identity = get_sliding_identity(seq1, seq2)
            if identity >= 0.30:
                adj[p1].add(p2)
                adj[p2].add(p1)
                
    # 2. Add edges for PDB co-occurrence in a pair
    for _, row in df_pairs.iterrows():
        p1, p2 = row['pdb1'], row['pdb2']
        adj[p1].add(p2)
        adj[p2].add(p1)
        
    print(f"Graph built in {time.time() - start_time:.4f}s")
    
    # Find connected components of PDBs
    visited = set()
    pdb_to_comp = {}
    comp_counter = 0
    
    for pdb in unique_pdbs:
        if pdb in visited:
            continue
        comp_id = f"comp_{comp_counter}"
        comp_counter += 1
        
        queue = [pdb]
        visited.add(pdb)
        while queue:
            curr = queue.pop(0)
            pdb_to_comp[curr] = comp_id
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
    pair_to_comp = []
    warnings_count = 0
    for idx, row in df_pairs.iterrows():
        c1 = pdb_to_comp[row['pdb1']]
        c2 = pdb_to_comp[row['pdb2']]
        if c1 != c2:
            warnings_count += 1
            if warnings_count <= 5:
                print(f"Warning: pair {idx} ({row['pdb1']}, {row['pdb2']}) has different components: {c1} vs {c2}")
        pair_to_comp.append(c1)
        
    print(f"Total different component warnings: {warnings_count}")
    
    df_pairs['family_id'] = pair_to_comp
    comp_counts = df_pairs['family_id'].value_counts()
    print(f"Found {len(comp_counts)} components.")
    print("Top 10 component sizes:")
    print(comp_counts.head(10))

if __name__ == "__main__":
    main()
