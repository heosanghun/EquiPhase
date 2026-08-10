import hashlib
import glob
import os

def generate_manifest():
    files_to_hash = glob.glob('claude_*.py') + ['z_inits_sealed.pt']
    manifest_content = ["Algorithm Hash                                                             Path",
                        "--------- ----                                                             ----"]
    
    for f_path in files_to_hash:
        if not os.path.isfile(f_path): continue
        with open(f_path, 'rb') as f:
            h = hashlib.sha256(f.read()).hexdigest().upper()
        abs_path = os.path.abspath(f_path)
        manifest_content.append(f"SHA256    {h} {abs_path}")
        
    with open('SEALED_MANIFEST_20260808.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(manifest_content))

if __name__ == "__main__":
    generate_manifest()
