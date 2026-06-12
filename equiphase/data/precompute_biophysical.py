import os
import pickle
import pandas as pd
import numpy as np
import sys

BASE_DIR = "D:/AI/EquiPhase/"
DATA_DIR = os.path.join(BASE_DIR, 'equiphase', 'data')
sys.path.append(BASE_DIR)
from equiphase.data.biophysical import extract_biophysical_features

def main():
    # Load all sequences from splits
    seqs = set()
    for split in ['train_phase5', 'val_phase5', 'test_phase5']:
        path = os.path.join(DATA_DIR, f"{split}.tsv")
        if os.path.exists(path):
            df = pd.read_csv(path, sep="\t")
            seqs.update(df['Sequence'].dropna().unique())
            
    seq_list = sorted(list(seqs))
    print(f"Total unique sequences to process: {len(seq_list)}")
    
    biophysical_dict = {}
    for seq in seq_list:
        biophysical_dict[seq] = extract_biophysical_features(seq)
        
    out_path = os.path.join(DATA_DIR, "biophysical_features.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(biophysical_dict, f)
        
    print(f"Successfully precomputed biophysical features for {len(biophysical_dict)} sequences and saved to {out_path}")

if __name__ == "__main__":
    main()
