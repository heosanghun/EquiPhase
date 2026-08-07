import os
import sys
import pandas as pd

val_tsv = r"C:\Project\AI\EquiPhase\equiphase\data\val.tsv"
df_val = pd.read_csv(val_tsv, sep="\t")

exact_unknown = 0
substring_unknown = 0

for idx, row in df_val.iterrows():
    raw_str = str(row['Sequence']).strip()
    raw_upper = raw_str.upper()
    
    if raw_upper == "UNKNOWN" or raw_upper == "N/A" or len(raw_str) == 0:
        exact_unknown += 1
    elif "UNKNOWN" in raw_upper:
        substring_unknown += 1
        print(f"Row {idx:>3d} has substring UNKNOWN: {row['Protein name']} | Snippet: {raw_str[:50]}")

print("=" * 80)
print(f"Exact match UNKNOWN / N/A rows: {exact_unknown} / {len(df_val)}")
print(f"Substring UNKNOWN (not exact) rows: {substring_unknown} / {len(df_val)}")
print("=" * 80)
