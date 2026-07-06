import os
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from tqdm import tqdm

print("Loading labels...")
df = pd.read_csv("data/wbc-bench-2026/phase1_label.csv")
image_dir = "data/wbc-bench-2026/phase1"

# Map to binary target: 1 if BL (blast cells), 0 otherwise
df['binary_label'] = (df['labels'] == 'BL').astype(int)

# Extract patient IDs (first 5 chars of filename)
df['patient_id'] = df['ID'].apply(lambda x: x[:5])

features = []
labels = []
groups = []
filenames = []

print("Extracting features (corners + center region) from images...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    img_name = row['ID']
    img_path = os.path.join(image_dir, img_name)
    if not os.path.exists(img_path):
        continue
    try:
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            w, h = img.size
            img_np = np.array(img) / 255.0
            
            # 1. Corner background features (average of 4 corners)
            corners = [
                img_np[0, 0],
                img_np[0, w - 1],
                img_np[h - 1, 0],
                img_np[h - 1, w - 1]
            ]
            avg_bg = np.mean(corners, axis=0) # shape (3,)
            
            # 2. Center cell features (center 50% region)
            w_start, w_end = int(w * 0.25), int(w * 0.75)
            h_start, h_end = int(h * 0.25), int(h * 0.75)
            center_region = img_np[h_start:h_end, w_start:w_end]
            
            avg_center = np.mean(center_region, axis=(0, 1)) # shape (3,)
            std_center = np.std(center_region, axis=(0, 1)) # shape (3,)
            
            # 3. Overall brightness
            avg_brightness = np.mean(img_np) # shape (1,)
            
            # Combine into 10-dimensional feature vector
            feat = np.concatenate([avg_bg, avg_center, std_center, [avg_brightness]])
            
            features.append(feat)
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

with open("data/wbc_features_bl.pkl", "wb") as f:
    pickle.dump(data, f)
print("Saved features to data/wbc_features_bl.pkl")
