import os
# Set Hugging Face mirror endpoint to bypass LFS CDN blocks
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

# Monkey-patch platform module to bypass Windows WMI query hangs
import platform
from collections import namedtuple
UnameResult = namedtuple('UnameResult', ['system', 'node', 'release', 'version', 'machine', 'processor'])
platform.win32_ver = lambda *args, **kwargs: ('10', '10.0.0', '', 'Multiprocessor Free')
platform.uname = lambda: UnameResult('Windows', 'DESKTOP-XXX', '10', '10.0.0', 'AMD64', 'AMD64')
platform.machine = lambda: 'AMD64'
platform.system = lambda: 'Windows'
platform.processor = lambda: 'AMD64'
platform.release = lambda: '10'
platform.version = lambda: '10.0.0'

import pickle
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel
from tqdm import tqdm

BASE_DIR = "D:/AI/EquiPhase/"
DATA_DIR = os.path.join(BASE_DIR, 'equiphase', 'data')
EMB_PATH = os.path.join(DATA_DIR, "esm2_embeddings.pkl")
LOCAL_MODEL_DIR = os.path.join(BASE_DIR, "models", "esm2_t33_650M_UR50D")

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
    
    print(f"Loading local model from {LOCAL_MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)
    model = EsmModel.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True).to(device)
    model.eval()
    
    if device.type == "cuda":
        model = model.half()
        print("Using half precision (float16) for ESM-2 650M")
        
    embeddings = {}
    batch_size = 16
    
    with torch.no_grad():
        for i in tqdm(range(0, len(seq_list), batch_size)):
            batch_seqs = seq_list[i:i+batch_size]
            inputs = tokenizer(batch_seqs, padding=True, truncation=True, return_tensors="pt").to(device)
            
            outputs = model(**inputs)
            attention_mask = inputs['attention_mask'] # (B, L)
            last_hidden = outputs.last_hidden_state # (B, L, D)
            
            for j, seq in enumerate(batch_seqs):
                seq_len = int(attention_mask[j].sum().item())
                if seq_len > 2:
                    emb = last_hidden[j, 1:seq_len-1].mean(dim=0).cpu().float().numpy()
                else:
                    emb = last_hidden[j].mean(dim=0).cpu().float().numpy()
                embeddings[seq] = emb
                
    with open(EMB_PATH, "wb") as f:
        pickle.dump(embeddings, f)
        
    print(f"Successfully saved {len(embeddings)} embeddings to {EMB_PATH}")

if __name__ == "__main__":
    main()
