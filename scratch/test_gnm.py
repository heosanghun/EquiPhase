import numpy as np
import sys
sys.path.append("D:/AI/EquiPhase")
from iss_data import parse_pdb

def compute_gnm_flexibility(coords, cutoff=10.0):
    L = coords.shape[0]
    if L <= 1:
        return 0.0
    # Construct distance matrix
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    Gamma = np.zeros((L, L))
    mask = (D < cutoff) & (~np.eye(L, dtype=bool))
    Gamma[mask] = -1.0
    for i in range(L):
        Gamma[i, i] = -np.sum(Gamma[i, :])
    
    # Pseudo-inverse
    Gamma_pinv = np.linalg.pinv(Gamma)
    # Average of diagonal (MSF)
    flexibility = np.mean(np.diag(Gamma_pinv))
    return flexibility

def main():
    path = "data/clean_chains/1aki_A.pdb"
    seq, coords = parse_pdb(path)
    if coords is not None:
        flex = compute_gnm_flexibility(coords)
        print(f"PDB {path} GNM flexibility: {flex:.6f}")
    else:
        print("Failed to parse PDB.")

if __name__ == "__main__":
    main()
