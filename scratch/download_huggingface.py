import os
import requests
import zipfile
from tqdm import tqdm

url = "https://huggingface.co/datasets/nttduc/wbc/resolve/main/wbc-bench-2026.zip"
dest_dir = "data"
zip_path = os.path.join(dest_dir, "wbc-bench-2026.zip")

os.makedirs(dest_dir, exist_ok=True)

print("Starting download from Hugging Face...")
response = requests.get(url, stream=True)
total_size = int(response.headers.get('content-length', 0))

with open(zip_path, "wb") as file, tqdm(
    desc="Downloading",
    total=total_size,
    unit='B',
    unit_scale=True,
    unit_divisor=1024,
) as bar:
    for data in response.iter_content(chunk_size=1024 * 1024):
        size = file.write(data)
        bar.update(size)

print("Download complete. Extracting files...")
extract_path = os.path.join(dest_dir, "wbc-bench-2026")
os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    file_list = zip_ref.namelist()
    for file in tqdm(file_list, desc="Extracting"):
        zip_ref.extract(file, extract_path)

print(f"Extraction complete. Files extracted to {extract_path}")
# Delete the zip file to save space
os.remove(zip_path)
print("Cleaned up zip file.")
