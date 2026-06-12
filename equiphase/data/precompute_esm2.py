import os
import pickle
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel
from tqdm import tqdm

BASE_DIR = "D:/AI/EquiPhase/"
DATA_DIR = os.path.join(BASE_DIR, 'equiphase', 'data')
EMB_PATH = os.path.join(DATA_DIR, "esm2_embeddings.pkl")

def main():
    # Load all sequences from splits
    seqs = set()
    for split in ['train', 'val', 'test']:
        path = os.path.join(DATA_DIR, f"{split}.tsv")
        if os.path.exists(path):
            df = pd.read_csv(path, sep="\t")
            seqs.update(df['Sequence'].dropna().unique())
            
    seq_list = sorted(list(seqs))
    print(f"Total unique sequences to precompute: {len(seq_list)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device)
    model.eval()
    
    embeddings = {}
    batch_size = 32
    
    with torch.no_grad():
        for i in tqdm(range(0, len(seq_list), batch_size)):
            batch_seqs = seq_list[i:i+batch_size]
            inputs = tokenizer(batch_seqs, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(**inputs)
            # Use mean pool of hidden states over length (excluding padding)
            attention_mask = inputs['attention_mask'] # (B, L)
            last_hidden = outputs.last_hidden_state # (B, L, D)
            
            for j, seq in enumerate(batch_seqs):
                seq_len = int(attention_mask[j].sum().item())
                # Exclude <cls> and <eos> tokens by taking average over content tokens
                # Usually esm tokenizer adds <cls> at index 0 and <eos> at the end
                if seq_len > 2:
                    emb = last_hidden[j, 1:seq_len-1].mean(dim=0).cpu().numpy()
                else:
                    emb = last_hidden[j].mean(dim=0).cpu().numpy()
                embeddings[seq] = emb
                
    with open(EMB_PATH, "wb") as f:
        pickle.dump(embeddings, f)
        
    print(f"Successfully saved {len(embeddings)} embeddings to {EMB_PATH}")

if __name__ == "__main__":
    main()
