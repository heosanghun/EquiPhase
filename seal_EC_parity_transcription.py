import re
import os
import hashlib
import sys
import platform

def get_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def main():
    print("SEAL_EC_PARITY_TRANSCRIPTION_BEGIN")
    print(f"Platform: {platform.platform()} | Python: {sys.version.split()[0]}")
    try:
        import torch
        print(f"Torch: {torch.__version__} | CUDA: {torch.version.cuda} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    except:
        pass

    # 1. CONFIG
    CONFIG = {
        "VANILLA_SCRIPT": "claude_paper2_baselines_sealed.py",
        "VANILLA_STDOUT": "base_run2_raw.txt",
        "MONOTONE_SCRIPT": "claude_paper2_baselines_sealed.py",
        "MONOTONE_STDOUT": "base_run2_raw.txt"
    }
    
    for k, v in CONFIG.items():
        if not os.path.exists(v):
            print(f"ABORT: Missing {k} at {v}")
            sys.exit(1)
        print(f"[{k}] {v} (SHA-256: {get_hash(v)[:8]}...)")

    # 2. EXTRACTION_SPECS
    EXTRACTION_SPECS = {
        "VANILLA": [
            ("Layer Width", r"fc1 = nn\.Linear\(.*?, (\d+)\)", "script"),
            ("Learning Rate", r"lr=([0-9eE\.\-]+)", "script"),
            ("Parameter Count", r"=== BASELINE 1.*?parameter count = (\d+)", "stdout")
        ],
        "MONOTONE": [
            ("Layer Width", r"W = nn\.Linear\(.*?, 2 \* (\w+)", "script"),
            ("Learning Rate", r"lr=([0-9eE\.\-]+)", "script"),
            ("Parameter Count", r"=== BASELINE 2.*?parameter count = (\d+)", "stdout"),
            ("Solver Max Iters", r"TRAIN_SOLVER_STEPS\s*=\s*(\d+)", "script"),
            ("Stopping Criterion (Residual)", r"TRAIN_SOLVER_STEPS\s*=\s*(\d+)", "script")
        ]
    }

    # 3. Extraction Logic
    def extract_values(model_name, specs, script_path, stdout_path):
        print(f"\n--- {model_name} PARITY FIELDS ---")
        with open(script_path, 'r', encoding='utf-8') as f:
            script_text = f.read()
        with open(stdout_path, 'r', encoding='utf-8') as f:
            stdout_text = f.read()
            
        for name, pattern, source in specs:
            text = script_text if source == "script" else stdout_text
            match = re.search(pattern, text, re.DOTALL)
            if match:
                print(f"{name:30s} : {match.group(1)}")
            else:
                print(f"ABORT: Failed to extract '{name}' using pattern '{pattern}' in {source}")
                sys.exit(1)

    extract_values("VANILLA", EXTRACTION_SPECS["VANILLA"], CONFIG["VANILLA_SCRIPT"], CONFIG["VANILLA_STDOUT"])
    extract_values("MONOTONE", EXTRACTION_SPECS["MONOTONE"], CONFIG["MONOTONE_SCRIPT"], CONFIG["MONOTONE_STDOUT"])
    
    script_hash = get_hash(__file__)
    print(f"\n[SELF] seal_EC_parity_transcription.py SHA-256: {script_hash}")
    print("SEAL_EC_PARITY_TRANSCRIPTION_END")

if __name__ == "__main__":
    main()
