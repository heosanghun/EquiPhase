import os
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from tqdm import tqdm

print("Loading labels...")
df = pd.read_csv("data/wbc-bench-2026/phase1_label.csv")
image_dir = "data/wbc-bench-2026/phase1"

# Map to binary target: 1 if SNE, 0 otherwise
df['binary_label'] = (df['labels'] == 'SNE').astype(int)

# Extract patient IDs (first 5 chars of filename)
df['patient_id'] = df['ID'].apply(lambda x: x[:5])

features = []
labels = []
groups = []
filenames = []

print("Extracting background features from images...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    img_name = row['ID']
    img_path = os.path.join(image_dir, img_name)
    if not os.path.exists(img_path):
        continue
    try:
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            w, h = img.size
            # Get corners
            corners = [
                img.getpixel((0, 0)),
                img.getpixel((w - 1, 0)),
                img.getpixel((0, h - 1)),
                img.getpixel((w - 1, h - 1))
            ]
            # Average color of corners
            avg_color = np.mean(corners, axis=0) / 255.0
            
            features.append(avg_color)
            labels.append(row['binary_label'])
            groups.append(row['patient_id'])
            filenames.append(img_name)
    except Exception as e:
        print(f"Error reading {img_name}: {e}")

features = np.array(features)
labels = np.array(labels)
groups = np.array(groups)

print(f"Extracted {len(features)} samples.")
print(f"Features shape: {features.shape}")

# Save to pickle
data = {
    "features": features,
    "labels": labels,
    "groups": groups,
    "filenames": filenames
}

with open("data/wbc_features.pkl", "wb") as f:
    pickle.dump(data, f)
print("Saved features to data/wbc_features.pkl")
