import os
import re
import pandas as pd
import numpy as np
import time

BASE_DIR = "D:/AI/EquiPhase/"
DATA_DIR = os.path.join(BASE_DIR, 'equiphase', 'data', 'raw')
OUTPUT_DIR = os.path.join(BASE_DIR, 'equiphase', 'data')

LLPS_POS_PATH = os.path.join(DATA_DIR, "Phase_separation_unambiguous", "Phase_separation_Unambiguous", "LLPS.xls")
LLPS_NEG_PATH = os.path.join(DATA_DIR, "No_phase_separation_unambiguous", "No_phase_separation_Unambiguous", "LLPS.xls")

def parse_range_mean(val_str):
    if pd.isna(val_str) or not isinstance(val_str, str):
        return None
    val_clean = val_str.strip().lower()
    match_range = re.search(r'([0-9.]+)\s*(?:-|to)\s*([0-9.]+)', val_clean)
    if match_range:
        try:
            return (float(match_range.group(1)) + float(match_range.group(2))) / 2.0
        except ValueError:
            pass
    match_val = re.search(r'([0-9.]+)', val_clean)
    if match_val:
        try:
            return float(match_val.group(1))
        except ValueError:
            pass
    return None

def parse_solute_concentration(val):
    if pd.isna(val) or not isinstance(val, str):
        return None
    val_clean = val.strip().lower()
    match_um = re.search(r'([0-9.]+(?:\s*(?:-|to)\s*[0-9.]+)?)\s*(?:um|uM|µm|µM|micromolar|μm|μM)', val_clean)
    if match_um:
        return parse_range_mean(match_um.group(1))
    match_mm = re.search(r'([0-9.]+(?:\s*(?:-|to)\s*[0-9.]+)?)\s*(?:mm|mM|millimolar)', val_clean)
    if match_mm:
        val_parsed = parse_range_mean(match_mm.group(1))
        return val_parsed * 1000.0 if val_parsed is not None else None
    match_mgml = re.search(r'([0-9.]+(?:\s*(?:-|to)\s*[0-9.]+)?)\s*(?:mg/ml|mg\s*ml-1)', val_clean)
    if match_mgml:
        val_parsed = parse_range_mean(match_mgml.group(1))
        return ("mg/ml", val_parsed)
    return None

def parse_salt_concentration(val):
    if pd.isna(val) or not isinstance(val, str):
        if isinstance(val, (int, float)) and not np.isnan(val):
            return float(val)
        return None
    val_clean = val.strip().lower()
    if any(term in val_clean for term in ['no salt', 'without salt', 'salt-free', '0 salt']):
        return 0.0
    match_mm = re.search(r'([0-9.]+(?:\s*(?:-|to)\s*[0-9.]+)?)\s*(?:mm|mM|millimolar)', val_clean)
    if match_mm:
        return parse_range_mean(match_mm.group(1))
    match_m = re.search(r'([0-9.]+(?:\s*(?:-|to)\s*[0-9.]+)?)\s*(?:\s+m|M|molar)\b', val_clean)
    if match_m:
        if not re.search(r'([0-9.]+)\s*mm', val_clean):
            val_parsed = parse_range_mean(match_m.group(1))
            return val_parsed * 1000.0 if val_parsed is not None else None
    match_num = re.search(r'([0-9.]+)', val_clean)
    if match_num:
        try:
            return float(match_num.group(1))
        except ValueError:
            pass
    return None

def parse_ph(val_buffer, val_salt):
    text = ""
    if isinstance(val_buffer, str):
        text += " " + val_buffer.lower()
    if isinstance(val_salt, str):
        text += " " + val_salt.lower()
    match = re.search(r'ph\s*[:=]?\s*([0-9.]+(?:\s*(?:-|to)\s*[0-9.]+)?)\b', text)
    if match:
        return parse_range_mean(match.group(1))
    return None

def parse_temp(val):
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)) and not np.isnan(val):
        return float(val)
    if not isinstance(val, str):
        return None
    val_clean = val.strip().lower()
    if any(term in val_clean for term in ['room temp', 'rt', 'room-temperature', 'ambient']):
        return 25.0
    match = re.search(r'([0-9.]+(?:\s*(?:-|to)\s*[0-9.]+)?)\s*(?:°c|celsius|c|\bc\b)?', val_clean)
    if match:
        return parse_range_mean(match.group(1))
    return None

def main():
    print("Loading datasets...")
    df_pos = pd.read_excel(LLPS_POS_PATH)
    df_neg = pd.read_excel(LLPS_NEG_PATH)
    
    df_pos['label'] = 1
    df_neg['label'] = 0
    df_all = pd.concat([df_pos, df_neg], ignore_index=True)
    
    # Process
    df_all['parsed_c_sat'] = df_all['Solute concentration'].apply(parse_solute_concentration)
    df_all['parsed_salt'] = df_all['Salt concentration'].apply(parse_salt_concentration)
    df_all['parsed_ph'] = df_all.apply(lambda row: parse_ph(row['Buffer'], row['Salt concentration']), axis=1)
    df_all['parsed_temp'] = df_all['Temperature'].apply(parse_temp)
    
    # We require parsed_c_sat to be numeric molarity (mg/ml has to be converted, or is filtered)
    # Let's check which records are complete
    df_all['is_complete'] = (
        df_all['Sequence'].notna() & 
        df_all['parsed_c_sat'].notna() & 
        df_all['parsed_c_sat'].apply(lambda x: isinstance(x, (int, float))) & 
        df_all['parsed_salt'].notna() & 
        df_all['parsed_ph'].notna() & 
        df_all['parsed_temp'].notna()
    )
    
    df_complete = df_all[df_all['is_complete']].copy()
    unique_seqs = df_complete['Sequence'].unique()
    print(f"Total complete records: {len(df_complete)}")
    print(f"Unique sequences in complete: {len(unique_seqs)}")
    
    # Jaccard 3-mer sequence clustering
    print("Clustering unique sequences...")
    kmers = [set(seq[i:i+3] for i in range(len(seq)-2)) for seq in unique_seqs]
    lengths = [len(seq) for seq in unique_seqs]
    
    sorted_idx = np.argsort(lengths)[::-1]
    
    clusters = []
    representatives = []
    
    for idx in sorted_idx:
        seq_kmers = kmers[idx]
        seq_len = lengths[idx]
        
        matched = False
        for rep_idx, rep in enumerate(representatives):
            rep_len = lengths[rep]
            if seq_len / rep_len < 0.5 or seq_len / rep_len > 2.0:
                continue
            rep_kmers = kmers[rep]
            union_size = len(seq_kmers.union(rep_kmers))
            jaccard = len(seq_kmers.intersection(rep_kmers)) / union_size if union_size > 0 else 0
            
            if jaccard >= 0.15:
                clusters[rep_idx].append(idx)
                matched = True
                break
                
        if not matched:
            representatives.append(idx)
            clusters.append([idx])
            
    print(f"Clustering complete. Total families (clusters): {len(representatives)}")
    
    # Map sequence to cluster ID
    seq_to_cluster = {}
    for rep_idx, cluster in enumerate(clusters):
        for idx in cluster:
            seq = unique_seqs[idx]
            seq_to_cluster[seq] = rep_idx
            
    df_complete['cluster_id'] = df_complete['Sequence'].map(seq_to_cluster)
    
    # Condition splits
    df_low = df_complete[df_complete['parsed_salt'] <= 150].copy()
    df_high = df_complete[df_complete['parsed_salt'] > 300].copy()
    
    print(f"Low salt records: {len(df_low)} (families: {df_low['cluster_id'].nunique()})")
    print(f"High salt records: {len(df_high)} (families: {df_high['cluster_id'].nunique()})")
    
    # For Low salt, we split into train/val using family-disjoint split
    low_salt_families = sorted(list(df_low['cluster_id'].unique()))
    np.random.seed(42)
    np.random.shuffle(low_salt_families)
    
    # 80/20 split of families
    split_idx = int(0.80 * len(low_salt_families))
    train_families = set(low_salt_families[:split_idx])
    val_families = set(low_salt_families[split_idx:])
    
    df_train = df_low[df_low['cluster_id'].isin(train_families)].copy()
    df_val = df_low[df_low['cluster_id'].isin(val_families)].copy()
    
    # High-salt goes directly to test
    df_test = df_high.copy()
    
    print("\n--- Split Statistics ---")
    print(f"Train set: {len(df_train)} records, {df_train['cluster_id'].nunique()} families, {df_train['label'].mean()*100:.1f}% positive")
    print(f"Val set: {len(df_val)} records, {df_val['cluster_id'].nunique()} families, {df_val['label'].mean()*100:.1f}% positive")
    print(f"Test set: {len(df_test)} records, {df_test['cluster_id'].nunique()} families, {df_test['label'].mean()*100:.1f}% positive")
    
    # Verify disjointness of train and val families
    overlap_train_val = set(df_train['cluster_id']).intersection(set(df_val['cluster_id']))
    print(f"Overlap families between train and val: {len(overlap_train_val)}")
    
    # Save splits
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_train.to_csv(os.path.join(OUTPUT_DIR, "train.tsv"), sep="\t", index=False)
    df_val.to_csv(os.path.join(OUTPUT_DIR, "val.tsv"), sep="\t", index=False)
    df_test.to_csv(os.path.join(OUTPUT_DIR, "test.tsv"), sep="\t", index=False)
    print("Splits successfully generated and saved to equiphase/data/")

if __name__ == "__main__":
    main()
