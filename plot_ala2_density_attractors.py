# -*- coding: utf-8 -*-
import hashlib
import os
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = os.path.join("data", "ala2", "alanine-dipeptide-3x250ns-backbone-dihedrals.npz")
OUT_PNG = os.path.join("paper3_iclr", "fig1_ala2_density_attractors.png")
os.makedirs("paper3_iclr", exist_ok=True)

npz = np.load(DATA_PATH)
X = np.concatenate([np.asarray(npz[k]) for k in sorted(npz.files)], axis=0)
phi_deg = np.degrees(X[:, 0])
psi_deg = np.degrees(X[:, 1])

fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
counts, xedges, yedges, im = ax.hist2d(phi_deg, psi_deg, bins=100, cmap='viridis', cmin=1)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Frame Density [PENDING: auditor check]')

# 5-seed EquiPhase attractors overlay
equiphase_seeds = [
    [(-72.82, 153.38), (-77.55, -14.97), (52.31, 32.76)],
    [(-73.44, 151.45), (-78.82, -13.79), (53.73, 31.82)],
    [(-72.41, 153.29), (-78.74, -15.04), (52.72, 31.85)],
    [(-73.15, 152.73), (-76.59, -14.89), (54.00, 30.56)],
    [(-74.01, 152.84), (-77.79, -14.67), (52.57, 30.42)],
]

for idx, seed_attrs in enumerate(equiphase_seeds):
    for p, s in seed_attrs:
        ax.scatter(p, s, color='red', s=40, edgecolors='white', zorder=5, alpha=0.8,
                   label='EquiPhase Attractors' if idx == 0 and p == seed_attrs[0][0] else "")

ax.set_xlabel('$\phi$ [degrees]')
ax.set_ylabel('$\psi$ [degrees]')
ax.set_xlim([-180, 180])
ax.set_ylim([-180, 180])
ax.set_title('Alanine Dipeptide MD Density & EquiPhase Attractors\n[PENDING: auditor check]')
ax.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_PNG)
plt.close()

h = hashlib.sha256(open(OUT_PNG, 'rb').read()).hexdigest().upper()
print(f"Generated {OUT_PNG} | SHA256={h}")
