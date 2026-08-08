import os
import numpy as np

DATA_PATH = os.path.join("data", "ala2", "alanine-dipeptide-3x250ns-backbone-dihedrals.npz")
data = np.load(DATA_PATH)
print("Keys in npz archive:", data.files)
