import pandas as pd
import os

def inspect_dataset():
    csv_path = "data/mutations.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    print(f"Total rows in mutations.csv: {len(df)}")
    print("Columns:", list(df.columns))
    print("\nUnique fold families:")
    print(df["fold_family_id"].value_counts())
    print("\nSample rows:")
    print(df.head(3))
    
    # Check pdb files in data/pdbs
    pdb_dir = "data/pdbs"
    if os.path.exists(pdb_dir):
        files = os.listdir(pdb_dir)
        print(f"\nTotal files in data/pdbs: {len(files)}")
        print("Sample files:", files[:10])
    else:
        print("\ndata/pdbs directory does not exist.")

if __name__ == "__main__":
    inspect_dataset()
