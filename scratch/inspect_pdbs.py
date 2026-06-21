import os

def inspect_pdbs():
    pdb_dir = "data/pdbs"
    if not os.path.exists(pdb_dir):
        print(f"Error: {pdb_dir} not found.")
        return
        
    files = os.listdir(pdb_dir)
    pdb_files = [f for f in files if f.startswith('pdb_')]
    print(f"Found {len(pdb_files)} files starting with 'pdb_'")
    if pdb_files:
        print("Sample files:", pdb_files[:15])

if __name__ == "__main__":
    inspect_pdbs()
