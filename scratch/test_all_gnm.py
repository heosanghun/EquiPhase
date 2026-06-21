import os
import time
import numpy as np
import sys
sys.path.append("D:/AI/EquiPhase")
from iss_data import parse_pdb

def compute_gnm_flexibility(coords, cutoff=10.0):
    L = coords.shape[0]
    if L <= 1:
        return 0.0
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    Gamma = np.zeros((L, L))
    mask = (D < cutoff) & (~np.eye(L, dtype=bool))
    Gamma[mask] = -1.0
    for i in range(L):
        Gamma[i, i] = -np.sum(Gamma[i, :])
    try:
        Gamma_pinv = np.linalg.pinv(Gamma)
        return np.mean(np.diag(Gamma_pinv))
    except:
        return 0.0

def main():
    pdb_dir = "data/clean_chains"
    files = [f for f in os.listdir(pdb_dir) if f.endswith(".pdb")]
    print(f"Found {len(files)} PDB files. Testing GNM...")
    
    start_all = time.time()
    for i, file in enumerate(files):
        path = os.path.join(pdb_dir, file)
        start = time.time()
        seq, coords = parse_pdb(path)
        if coords is not None:
            L = coords.shape[0]
            flex = compute_gnm_flexibility(coords)
            dur = time.time() - start
            if dur > 0.1:
                print(f"[{i+1}/{len(files)}] {file} (L={L}) took {dur:.4f}s - flexibility={flex:.6f}")
        else:
            print(f"[{i+1}/{len(files)}] {file} failed to parse.")
            
    print(f"Total time for all GNM: {time.time() - start_all:.2f}s")

if __name__ == "__main__":
    main()
