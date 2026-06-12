import os
import requests
from tqdm import tqdm

BASE_DIR = "D:/AI/EquiPhase/"
LOCAL_MODEL_DIR = os.path.join(BASE_DIR, "models", "esm2_t33_650M_UR50D")
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# List of files to download
files = [
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json"
]

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, stream=True, verify=False)
    total_size = int(response.headers.get('content-length', 0))
    
    block_size = 1024 * 1024 # 1MB
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(dest_path, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    
    if total_size != 0 and t.n != total_size:
        print("ERROR: Something went wrong with the download size.")

def main():
    base_url = "https://hf-mirror.com/facebook/esm2_t33_650M_UR50D/resolve/main/"
    
    for f in files:
        url = base_url + f
        dest = os.path.join(LOCAL_MODEL_DIR, f)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"File {f} already exists and is not empty. Skipping.")
            continue
        download_file(url, dest)
        
    print(f"All files downloaded successfully to {LOCAL_MODEL_DIR}")

if __name__ == "__main__":
    main()
