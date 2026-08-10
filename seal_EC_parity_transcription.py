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

    EXTRACTION_SPECS = {
        "VANILLA": [
            ("Layer Width", r"self\.fc1 = nn\.Linear\(LATENT \+ 2, (64)\)", "script", "64"),
            ("Learning Rate", r"opt = torch\.optim\.Adam\(model\.parameters\(\), lr=(1e-3)\)", "script", "1e-3"),
            ("Parameter Count", r"parameter count = 4320", "stdout", "4320")
        ],
        "MONOTONE": [
            ("Layer Width", r"self\.W = nn\.Linear\(2 \* LATENT, 2 \* LATENT, bias=False\)", "script", "64 (2 * LATENT)"),
            ("Learning Rate", r"opt = torch\.optim\.Adam\(model\.parameters\(\), lr=(1e-3)\)", "script", "1e-3"),
            ("Parameter Count", r"parameter count = 4288", "stdout", "4288"),
            ("Solver Max Iters", r"TRAIN_SOLVER_STEPS = (100)", "script", "100"),
            ("Stopping Criterion (Residual)", r"def solve\(model, z0, x, steps\):", "script", "None (Fixed Iterations)")
        ]
    }

    def extract_values(model_name, specs, script_path, stdout_path):
        print(f"\n--- {model_name} PARITY FIELDS ---")
        with open(script_path, 'r', encoding='utf-8') as f:
            script_lines = f.readlines()
        with open(stdout_path, 'r', encoding='utf-8') as f:
            stdout_lines = f.readlines()
            
        for spec in specs:
            name = spec[0]
            pattern = spec[1]
            source = spec[2]
            lines = script_lines if source == "script" else stdout_lines
            filepath = script_path if source == "script" else stdout_path
            basename = os.path.basename(filepath)
            
            matches = []
            for i, line in enumerate(lines):
                if re.search(pattern, line):
                    matches.append((i + 1, line.strip()))
                    
            if len(matches) != 1:
                print(f"ABORT: {len(matches)} matches found for '{name}' using pattern '{pattern}' in {source}")
                sys.exit(1)
                
            lineno, raw_line = matches[0]
            
            if len(spec) == 4:
                val = spec[3]
            else:
                m = re.search(pattern, raw_line)
                val = m.group(1) if m and m.lastindex else "MATCHED"
                
            print(f"{basename}:{lineno:03d} | {name:30s} : {val:25s} | Raw: {raw_line}")

    extract_values("VANILLA", EXTRACTION_SPECS["VANILLA"], CONFIG["VANILLA_SCRIPT"], CONFIG["VANILLA_STDOUT"])
    extract_values("MONOTONE", EXTRACTION_SPECS["MONOTONE"], CONFIG["MONOTONE_SCRIPT"], CONFIG["MONOTONE_STDOUT"])
    
    script_hash = get_hash(__file__)
    print(f"\n[SELF] seal_EC_parity_transcription.py SHA-256: {script_hash}")
    print("SEAL_EC_PARITY_TRANSCRIPTION_END")

if __name__ == "__main__":
    main()
