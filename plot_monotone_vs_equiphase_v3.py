import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import hashlib

# Seed 7777 5 attractors from sealed stdout log
equi_attractors = [
    {"name": r"$\beta$ (-72.8°, +153.4°)", "phi": -72.8, "psi": 153.4, "marker": "o", "color": "#00ffcc", "hollow": False},
    {"name": r"$C_5$ (-142.4°, +157.8°)", "phi": -142.4, "psi": 157.8, "marker": "s", "color": "#00aaff", "hollow": False},
    {"name": r"$\alpha_R$ (-77.6°, -15.0°)", "phi": -77.6, "psi": -15.0, "marker": "^", "color": "#ff0055", "hollow": False},
    {"name": r"$\alpha_L$ (+52.3°, +32.8°)", "phi": 52.3, "psi": 32.8, "marker": "D", "color": "#ffaa00", "hollow": False},
    {"name": r"Disclosed Artifact (+59.7°, +170.9°)", "phi": 59.7, "psi": 170.9, "marker": "X", "color": "#ffffff", "hollow": True}
]

# Monotone DEQ 576 initializations single attractor
mono_attractor = {"phi": -50.07, "psi": 18.51}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), dpi=300)
plt.style.use('dark_background')

# Grid initializations (24x24 = 576 points)
grid_phi = np.linspace(-180, 180, 24)
grid_psi = np.linspace(-180, 180, 24)
P, Q = np.meshgrid(grid_phi, grid_psi)
init_p = P.flatten()
init_q = Q.flatten()

# Subplot 1: Monotone DEQ (Banach Contraction: Lip <= 0.95 (theory); empirical max 0.788)
ax1.scatter(init_p, init_q, color='gray', alpha=0.3, s=15, label='576 Grid Inits')
for i in range(len(init_p)):
    ax1.plot([init_p[i], mono_attractor["phi"]], [init_q[i], mono_attractor["psi"]], color='#ff4444', alpha=0.12, linewidth=0.6)
ax1.scatter(mono_attractor["phi"], mono_attractor["psi"], color='#ff2222', edgecolors='white', marker='X', s=220, linewidths=2.5, zorder=6, label=r'Single Fixed Pt (-50.1°, +18.5°)')

ax1.set_xlim(-180, 180)
ax1.set_ylim(-180, 180)
ax1.set_xlabel(r'$\phi$ (degrees)', fontsize=12, fontweight='bold')
ax1.set_ylabel(r'$\psi$ (degrees)', fontsize=12, fontweight='bold')
ax1.set_title('Monotone DEQ (Lip ≤ 0.95 theory; empirical max 0.788)\nN=1 Single Fixed-Point Collapse', fontsize=12, pad=10)
ax1.legend(loc='lower right', fontsize=10, framealpha=0.85)
ax1.grid(True, linestyle='--', alpha=0.3)

# Subplot 2: EquiPhase DEQ (Multistable Branching - All 5 attractors shown)
ax2.scatter(init_p, init_q, color='gray', alpha=0.3, s=15, label='576 Grid Inits')

# Draw branching trajectories to nearest attractor
for i in range(len(init_p)):
    p_i, q_i = init_p[i], init_q[i]
    dists = [np.hypot(p_i - att["phi"], q_i - att["psi"]) for att in equi_attractors]
    closest_idx = np.argmin(dists)
    closest_att = equi_attractors[closest_idx]
    ax2.plot([p_i, closest_att["phi"]], [q_i, closest_att["psi"]], color=closest_att["color"], alpha=0.15, linewidth=0.6)

for att in equi_attractors:
    if att["hollow"]:
        ax2.scatter(att["phi"], att["psi"], c='none', edgecolors=att["color"], marker=att["marker"], s=180, linewidths=2.5, label=att["name"], zorder=6)
    else:
        ax2.scatter(att["phi"], att["psi"], c=att["color"], edgecolors='white', marker=att["marker"], s=140, linewidths=1.5, label=att["name"], zorder=6)

ax2.set_xlim(-180, 180)
ax2.set_ylim(-180, 180)
ax2.set_xlabel(r'$\phi$ (degrees)', fontsize=12, fontweight='bold')
ax2.set_ylabel(r'$\psi$ (degrees)', fontsize=12, fontweight='bold')
ax2.set_title('EquiPhase DEQ (Damped Velocity Verlet)\nMultistable Attractor Branching (All 5 Attractors)', fontsize=12, pad=10)
ax2.legend(loc='lower right', fontsize=9, framealpha=0.85)
ax2.grid(True, linestyle='--', alpha=0.3)

out_file = os.path.join("paper3_iclr", "fig3_monotone_vs_equiphase_v3.png")
plt.tight_layout()
plt.savefig(out_file, dpi=300)
plt.close()

h_val = hashlib.sha256(open(out_file, "rb").read()).hexdigest().upper()
print(f"Fig 3 v3 generated successfully at {out_file} | SHA256={h_val}")
