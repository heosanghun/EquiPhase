import pandas as pd
import numpy as np

def main():
    df = pd.read_csv("honest_audit_results.csv")
    print("=== Unique Models in CSV ===")
    print(df["model"].unique())
    print("\n=== Unique Splits ===")
    print(df["split"].unique())
    
    # Filter for Placebo DEQ in HELD-OUT (H)
    df_placebo = df[(df["model"] == "Placebo DEQ") & (df["split"] == "HELD-OUT (H)")]
    
    print(f"\nPlacebo DEQ HELD-OUT (H) count: {len(df_placebo)}")
    if len(df_placebo) > 0:
        print("\n=== Averages for Placebo DEQ ===")
        print("mA Mean:", df_placebo["mA"].mean())
        print("mB Mean:", df_placebo["mB"].mean())
        print("mDecoy Mean:", df_placebo["mDecoy"].mean())
        
        # Calculate pair margin
        df_placebo = df_placebo.copy()
        df_placebo["m_pair"] = np.minimum(df_placebo["mA"], df_placebo["mB"])
        
        # Split into switchers vs controls
        sw = df_placebo[df_placebo["is_switcher"] == 1]
        ctrl = df_placebo[df_placebo["is_switcher"] == 0]
        
        print("\n=== Switchers vs Controls Margins ===")
        print(f"Switcher (n={len(sw)}) | mA Mean: {sw['mA'].mean():.6f} | mB Mean: {sw['mB'].mean():.6f} | m_pair Mean: {sw['m_pair'].mean():.6f}")
        print(f"Control  (n={len(ctrl)}) | mA Mean: {ctrl['mA'].mean():.6f} | mB Mean: {ctrl['mB'].mean():.6f} | m_pair Mean: {ctrl['m_pair'].mean():.6f}")
        
        # Check standard deviation of margins
        print("\n=== Margin Std Dev ===")
        print("mA Std:", df_placebo["mA"].std())
        print("mB Std:", df_placebo["mB"].std())
        print("m_pair Std:", df_placebo["m_pair"].std())

if __name__ == "__main__":
    main()
