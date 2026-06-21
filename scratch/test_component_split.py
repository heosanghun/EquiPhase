import pandas as pd
import numpy as np
import os
import sys
sys.path.append("D:/AI/EquiPhase")
from iss_data import parse_pdb

def main():
    df_pairs = pd.read_csv("data/benchmark_pairs.csv")
    unique_pdbs = list(set(df_pairs['pdb1'].tolist() + df_pairs['pdb2'].tolist()))
    
    parsed_seqs = {}
    for pdb in unique_pdbs:
        path = f"data/clean_chains/{pdb}.pdb"
        seq, _ = parse_pdb(path)
        parsed_seqs[pdb] = seq
        
    # Build a graph of PDBs based on 30% identity
    print("Building sequence identity graph for PDBs...")
    adj = {pdb: set() for pdb in unique_pdbs}
    for i, p1 in enumerate(unique_pdbs):
        seq1 = parsed_seqs[p1]
        for j in range(i + 1, len(unique_pdbs)):
            p2 = unique_pdbs[j]
            seq2 = parsed_seqs[p2]
            
            len_ratio = len(seq1) / len(seq2)
            if len_ratio < 0.5 or len_ratio > 2.0:
                continue
                
            min_len = min(len(seq1), len(seq2))
            mismatches = sum(1 for a, b in zip(seq1[:min_len], seq2[:min_len]) if a != b)
            identity = (min_len - mismatches) / max(len(seq1), len(seq2))
            
            if identity >= 0.30:
                adj[p1].add(p2)
                adj[p2].add(p1)
                
    # Now build connected components of PDBs
    visited = set()
    pdb_to_comp = {}
    comp_counter = 0
    
    for pdb in unique_pdbs:
        if pdb in visited:
            continue
        comp_id = f"comp_{comp_counter}"
        comp_counter += 1
        
        # BFS
        queue = [pdb]
        visited.add(pdb)
        while queue:
            curr = queue.pop(0)
            pdb_to_comp[curr] = comp_id
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
    # Now group pairs by their component
    # A pair contains pdb1 and pdb2. Since pdb1 and pdb2 are always the same protein (or variants),
    # they must belong to the same sequence identity component. Let's verify.
    pair_to_comp = []
    for idx, row in df_pairs.iterrows():
        c1 = pdb_to_comp[row['pdb1']]
        c2 = pdb_to_comp[row['pdb2']]
        if c1 != c2:
            print(f"Warning: pair {idx} has different components for pdb1 ({c1}) and pdb2 ({c2})")
        pair_to_comp.append(c1)
        
    df_pairs['family_id'] = pair_to_comp
    comp_counts = df_pairs['family_id'].value_counts()
    print(f"\nFound {len(comp_counts)} components.")
    print("Component size distribution:")
    print(comp_counts.head(20))
    print(f"Total pairs: {len(df_pairs)}")
    
if __name__ == "__main__":
    main()
