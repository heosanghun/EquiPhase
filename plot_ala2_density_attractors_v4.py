import os
import numpy as np
import matplotlib.pyplot as plt
import hashlib

DATA_PATH = os.path.join("data", "ala2", "alanine-dipeptide-3x250ns-backbone-dihedrals.npz")
data = np.load(DATA_PATH)
traj0 = data['arr_0']
traj1 = data['arr_1']
traj2 = data['arr_2']
dihedrals = np.vstack([traj0, traj1, traj2])  # (750000, 2) in radians
phi = dihedrals[:, 0] * 180.0 / np.pi
psi = dihedrals[:, 1] * 180.0 / np.pi

# Seed 7777 5 attractors from sealed stdout log
attractors = [
    {"name": r"$\beta$ (-72.8°, +153.4°)", "phi": -72.8, "psi": 153.4, "marker": "o", "color": "#00ffcc", "hollow": False},
    {"name": r"$C_5$ (-142.4°, +157.8°)", "phi": -142.4, "psi": 157.8, "marker": "s", "color": "#00aaff", "hollow": False},
    {"name": r"$\alpha_R$ (-77.6°, -15.0°)", "phi": -77.6, "psi": -15.0, "marker": "^", "color": "#ff0055", "hollow": False},
    {"name": r"$\alpha_L$ (+52.3°, +32.8°)", "phi": 52.3, "psi": 32.8, "marker": "D", "color": "#ffaa00", "hollow": False},
    {"name": r"Disclosed Artifact (+59.7°, +170.9°)", "phi": 59.7, "psi": 170.9, "marker": "X", "color": "#ffffff", "hollow": True}
]

fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
plt.style.use('dark_background')

# 2D Density histogram
h, xedges, yedges = np.histogram2d(phi, psi, bins=120, range=[[-180, 180], [-180, 180]])
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
log_h = np.log10(h.T + 1e-4)

im = ax.imshow(log_h, extent=extent, origin='lower', cmap='magma', aspect='equal')
cb = fig.colorbar(im, ax=ax, label=r'Log$_{10}$ Frame Density (750k frames)')

# Plot 5 attractors
for att in attractors:
    if att["hollow"]:
        ax.scatter(att["phi"], att["psi"], c='none', edgecolors=att["color"], marker=att["marker"], s=180, linewidths=2.5, label=att["name"], zorder=5)
    else:
        ax.scatter(att["phi"], att["psi"], c=att["color"], edgecolors='white', marker=att["marker"], s=140, linewidths=1.5, label=att["name"], zorder=5)

ax.set_xlim(-180, 180)
ax.set_ylim(-180, 180)
ax.set_xticks(np.arange(-150, 180, 50))
ax.set_yticks(np.arange(-150, 180, 50))
ax.set_xlabel(r'$\phi$ (degrees)', fontsize=13, fontweight='bold', labelpad=8)
ax.set_ylabel(r'$\psi$ (degrees)', fontsize=13, fontweight='bold', labelpad=8)
ax.set_title(r'Alanine Dipeptide MD Density & EquiPhase Attractors (Seed 7777)', fontsize=13, pad=12)
ax.legend(loc='lower right', framealpha=0.85, fontsize=10)
ax.grid(True, linestyle='--', alpha=0.3)

out_file = os.path.join("paper3_iclr", "fig1_ala2_density_attractors_v4.png")
plt.tight_layout()
plt.savefig(out_file, dpi=300)
plt.close()

h_val = hashlib.sha256(open(out_file, "rb").read()).hexdigest().upper()
print(f"Fig 1 v4 generated successfully at {out_file} | SHA256={h_val}")
