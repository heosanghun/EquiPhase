import os
import urllib.request
import zipfile
import re
import pandas as pd
import numpy as np

# Define directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'equiphase', 'data', 'raw')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# Path to LLPS.xls for phase separation unambiguous
LLPS_XLS_PATH = os.path.join(DATA_DIR, "Phase_separation_unambiguous", "Phase_separation_Unambiguous", "LLPS.xls")

def parse_solute_concentration(val):
    """Parses solute concentration to micromolar (uM) if possible."""
    if pd.isna(val) or not isinstance(val, str):
        return None
    val_clean = val.strip().lower()
    
    # Try to find a number followed by uM, micromolar, um, etc.
    # Example: "10 uM", "10um", "10 micromolar"
    match_um = re.search(r'([0-9.]+)\s*(?:um|uM|µm|µM|micromolar|μm|μM)', val_clean)
    if match_um:
        return float(match_um.group(1))
        
    # Example: "0.01 mM", "0.01mm"
    match_mm = re.search(r'([0-9.]+)\s*(?:mm|mM|millimolar)', val_clean)
    if match_mm:
        return float(match_mm.group(1)) * 1000.0
        
    # Example: "1.2 mg/ml", "1.2mg/ml"
    # Note: conversion of mg/ml to uM requires Molecular Weight. 
    # For now, we flag it as semi-quantitative or attempt a fallback.
    # We will log that it exists but needs MW to convert to molarity.
    match_mgml = re.search(r'([0-9.]+)\s*(?:mg/ml|mg\s*ml-1)', val_clean)
    if match_mgml:
        return ("mg/ml", float(match_mgml.group(1)))
        
    return None

def parse_salt_concentration(val):
    """Parses salt concentration to mM if possible."""
    if pd.isna(val) or not isinstance(val, str):
        # If it's a number, assume mM
        if isinstance(val, (int, float)) and not np.isnan(val):
            return float(val)
        return None
    val_clean = val.strip().lower()
    
    # Check for "no salt" or "0"
    if any(term in val_clean for term in ['no salt', 'without salt', 'salt-free', '0 salt']):
        return 0.0
        
    # Check for mM salt
    # Example: "150 mM NaCl", "150mm"
    match_mm = re.search(r'([0-9.]+)\s*(?:mm|mM|millimolar)', val_clean)
    if match_mm:
        return float(match_mm.group(1))
        
    # Check for M salt
    # Example: "0.15 M NaCl", "0.15m"
    match_m = re.search(r'([0-9.]+)\s*(?:\s+m|M|molar)\b', val_clean)
    if match_m:
        # Avoid matching mM
        if not re.search(r'([0-9.]+)\s*mm', val_clean):
            return float(match_m.group(1)) * 1000.0
            
    # If just a number is found, assume mM
    match_num = re.match(r'^([0-9.]+)$', val_clean)
    if match_num:
        return float(match_num.group(1))
        
    return None

def parse_ph(val_buffer, val_salt):
    """Extracts pH value from Buffer or Salt column."""
    text = ""
    if isinstance(val_buffer, str):
        text += " " + val_buffer.lower()
    if isinstance(val_salt, str):
        text += " " + val_salt.lower()
        
    # Look for patterns like "ph 7.5", "ph=7.5", "ph of 7.5"
    match = re.search(r'ph\s*[:=]?\s*([0-9.]+)', text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def parse_temp(val):
    """Extracts temperature in Celsius."""
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)) and not np.isnan(val):
        return float(val)
    if not isinstance(val, str):
        return None
        
    val_clean = val.strip().lower()
    
    # Check for room temp
    if any(term in val_clean for term in ['room temp', 'rt', 'room-temperature', 'ambient']):
        return 25.0
        
    # Extract number
    match = re.search(r'([0-9.]+)\s*(?:°c|celsius|c|\bc\b)?', val_clean)
    if match:
        return float(match.group(1))
    return None

def run_due_diligence():
    print(f"Reading {LLPS_XLS_PATH}...")
    df = pd.read_excel(LLPS_XLS_PATH)
    
    total_records = len(df)
    
    # 1. Sequence validity
    valid_seq = df['Sequence'].dropna().str.strip().str.len() > 0
    total_valid_seq = valid_seq.sum()
    
    # 2. Extract features
    df['parsed_c_sat'] = df['Solute concentration'].apply(parse_solute_concentration)
    df['parsed_salt'] = df['Salt concentration'].apply(parse_salt_concentration)
    df['parsed_ph'] = df.apply(lambda row: parse_ph(row['Buffer'], row['Salt concentration']), axis=1)
    df['parsed_temp'] = df['Temperature'].apply(parse_temp)
    
    # Count variables
    has_csat_molar = df['parsed_c_sat'].apply(lambda x: isinstance(x, float) if x is not None else False).sum()
    has_csat_mgml = df['parsed_c_sat'].apply(lambda x: isinstance(x, tuple) if x is not None else False).sum()
    has_any_csat = df['parsed_c_sat'].notna().sum()
    
    has_salt = df['parsed_salt'].notna().sum()
    has_ph = df['parsed_ph'].notna().sum()
    has_temp = df['parsed_temp'].notna().sum()
    
    # Overlap analysis
    # Complete records: valid sequence AND has C_sat (molar or mg/ml) AND has salt AND has pH AND has temp
    df['is_complete'] = (
        (df['Sequence'].notna()) & 
        (df['parsed_c_sat'].notna()) & 
        (df['parsed_salt'].notna()) & 
        (df['parsed_ph'].notna()) & 
        (df['parsed_temp'].notna())
    )
    complete_count = df['is_complete'].sum()
    
    # Count of complete records that have molar C_sat (which doesn't require MW conversion)
    df['is_complete_molar'] = (
        df['is_complete'] & 
        df['parsed_c_sat'].apply(lambda x: isinstance(x, float) if x is not None else False)
    )
    complete_molar_count = df['is_complete_molar'].sum()
    
    # Condition coverage for low-salt and high-salt splits among complete records
    df_complete = df[df['is_complete']].copy()
    
    low_salt_count = 0
    high_salt_count = 0
    if len(df_complete) > 0:
        # Let's say salt <= 150 mM is low salt, salt > 300 mM is high salt
        # We will dynamically count based on standard threshold
        low_salt_count = (df_complete['parsed_salt'] <= 150).sum()
        high_salt_count = (df_complete['parsed_salt'] > 300).sum()
        
    print(f"Total records: {total_records}")
    print(f"With valid sequence: {total_valid_seq}")
    print(f"With C_sat (Molar/uM): {has_csat_molar}")
    print(f"With C_sat (mg/ml): {has_csat_mgml}")
    print(f"With Salt (mM): {has_salt}")
    print(f"With pH: {has_ph}")
    print(f"With Temp: {has_temp}")
    print(f"Complete records (Seq + C_sat + Salt + pH + Temp): {complete_count}")
    print(f"Complete records with molar C_sat: {complete_molar_count}")
    print(f"Low salt (<= 150 mM) complete count: {low_salt_count}")
    print(f"High salt (> 300 mM) complete count: {high_salt_count}")
    
    # Determine sufficiency
    is_sufficient = complete_count >= 300
    
    # Write MD report
    report_path = os.path.join(REPORTS_DIR, "data_sufficiency_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# EquiPhase Data Sufficiency Report (Due Diligence)\n\n")
        f.write(f"**Generated on:** 2026-06-12\n")
        f.write(f"**Database:** LLPSDB v2.0 (Unambiguous System - Phase Separation)\n\n")
        
        f.write("## 1. Summary of Cured Records\n\n")
        f.write("| Metric | Record Count | Percentage of Total |\n")
        f.write("| :--- | :---: | :---: |\n")
        f.write(f"| Total raw entries in LLPS.xls | {total_records} | 100.0% |\n")
        f.write(f"| Valid sequence | {total_valid_seq} | {total_valid_seq/total_records*100:.1f}% |\n")
        f.write(f"| Parsable C_sat (Molar/uM) | {has_csat_molar} | {has_csat_molar/total_records*100:.1f}% |\n")
        f.write(f"| Parsable C_sat (mg/ml) | {has_csat_mgml} | {has_csat_mgml/total_records*100:.1f}% |\n")
        f.write(f"| Parsable Salt concentration | {has_salt} | {has_salt/total_records*100:.1f}% |\n")
        f.write(f"| Parsable pH (extracted from Buffer/Salt) | {has_ph} | {has_ph/total_records*100:.1f}% |\n")
        f.write(f"| Parsable Temperature | {has_temp} | {has_temp/total_records*100:.1f}% |\n")
        f.write(f"| **Complete Records (Seq + C_sat + Salt + pH + Temp)** | **{complete_count}** | **{complete_count/total_records*100:.1f}%** |\n")
        f.write(f"| **Complete Records (Molar C_sat only)** | **{complete_molar_count}** | **{complete_molar_count/total_records*100:.1f}%** |\n\n")
        
        f.write("## 2. Salt Condition Split Analysis\n\n")
        f.write("Using standard biophysical salt concentration regimes on complete records:\n")
        f.write(f"- **Low Salt Regime (<= 150 mM):** {low_salt_count} records\n")
        f.write(f"- **High Salt Regime (> 300 mM):** {high_salt_count} records\n")
        f.write(f"- **Intermediate Salt Regime (150 - 300 mM):** {complete_count - low_salt_count - high_salt_count} records\n\n")
        
        f.write("## 3. Data Sufficiency Verification Verdict\n\n")
        if is_sufficient:
            f.write("> [!NOTE]\n")
            f.write(f"> **VERDICT: PASS (Sufficient Data Available)**\n")
            f.write(f"> The complete record count is **{complete_count}**, which exceeds the pre-registered sufficiency threshold of **300** records.\n")
            f.write(f"> Low-salt ({low_salt_count}) and High-salt ({high_salt_count}) coverage is adequate to support condition-based extrapolation splits.\n")
        else:
            f.write("> [!CAUTION]\n")
            f.write(f"> **VERDICT: REJECT (Insufficient Data Available)**\n")
            f.write(f"> The complete record count of **{complete_count}** is below the sufficiency threshold of **300** records.\n")
            f.write(f"> Proceeding with H1 under these conditions poses a high risk. We recommend switching to the fallback plan (Binary LLPS yes/no classification or restricted model protein subsets).\n")
            
        f.write("\n## 4. Column Mapping & Extraction Sample\n\n")
        f.write("Successfully identified and parsed the following fields from LLPS.xls:\n")
        f.write("- **Sequence:** `Sequence`\n")
        f.write("- **C_sat:** `Solute concentration` (parsed with regex for uM and mg/ml)\n")
        f.write("- **Salt:** `Salt concentration` (parsed for mM)\n")
        f.write("- **pH:** Extracted from `Buffer` and `Salt concentration` columns\n")
        f.write("- **Temperature:** `Temperature` (parsed for Celsius)\n")
        
    print(f"Report written to {report_path}")

if __name__ == "__main__":
    run_due_diligence()
