import os
import mdshare

target_dir = r"C:\Project\EquiPhase\data\ala2"
os.makedirs(target_dir, exist_ok=True)

files_to_fetch = [
    'alanine-dipeptide-nowater.pdb',
    'alanine-dipeptide-3x250ns-backbone-dihedrals.npz',
    'alanine-dipeptide-0-250ns-nowater.xtc'
]

for fname in files_to_fetch:
    print(f"Fetching {fname}...")
    fpath = mdshare.fetch(fname, working_directory=target_dir)
    print(f"Successfully downloaded {fname} -> {fpath}")

print("All Alanine Dipeptide files downloaded successfully.")
