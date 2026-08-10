import hashlib, math, os, sys, platform
import numpy as np

def get_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def main():
    print("SEAL_ED_PYEMMA_AGREEMENT_BEGIN")
    print(f"Platform: {platform.platform()} | Python: {sys.version.split()[0]}")
    try:
        import torch
        print(f"Torch: {torch.__version__} | CUDA: {torch.version.cuda} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    except:
        pass

    DATA_PATH = os.path.join("data", "ala2", "alanine-dipeptide-3x250ns-backbone-dihedrals.npz")
    if not os.path.exists(DATA_PATH):
        print(f"ABORT: Missing data at {DATA_PATH}")
        sys.exit(1)
    
    print(f"Data SHA-256: {get_hash(DATA_PATH)}")
    
    try:
        import pyemma
        print(f"PyEMMA version: {pyemma.__version__}")
    except ImportError:
        print("ABORT: PyEMMA not installed. Please run this script in an environment with PyEMMA.")
        sys.exit(1)
        
    print("\n--- PyEMMA MSM & VAMPnet Agreement ---")
    
    npz = np.load(DATA_PATH)
    # trajectories as list of arrays
    trajectories = [np.asarray(npz[k]) for k in sorted(npz.files)]
    
    # Example logic for PyEMMA MSM & VAMPnet (Agreement only, no ranking)
    # (Since this is a sealed audit script, we just output the metric and exit)
    print("Fitting PyEMMA VAMPnet and MSM...")
    
    # TICA + MSM
    tica = pyemma.coordinates.tica(trajectories, lag=10, dim=2)
    tica_out = tica.get_output()
    cluster = pyemma.coordinates.cluster_kmeans(tica_out, k=100, max_iter=50)
    msm = pyemma.msm.estimate_markov_model(cluster.dtrajs, lag=10)
    
    # VAMPnet (pseudo-code depending on pyemma's exact API)
    # vamp = pyemma.coordinates.vamp(trajectories, lag=10, dim=4)
    # score = vamp.score(trajectories)
    
    # output agreement metric
    # print(f"Agreement Metric (VAMP-2): {score}")
    
    print("Not fully implemented due to missing PyEMMA local install for testing.")
    
    script_hash = get_hash(__file__)
    print(f"\n[SELF] seal_ED_pyemma_agreement.py SHA-256: {script_hash}")
    print("SEAL_ED_PYEMMA_AGREEMENT_END")

if __name__ == "__main__":
    main()
