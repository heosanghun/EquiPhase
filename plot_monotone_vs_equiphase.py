# -*- coding: utf-8 -*-
import hashlib
import os
import numpy as np
import matplotlib.pyplot as plt

OUT_PNG = os.path.join("paper3_iclr", "fig3_monotone_vs_equiphase.png")
os.makedirs("paper3_iclr", exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

# Panel 1: Monotone DEQ (N=1 Collapse)
# 576 initial grid points all collapsing to (-50.07, +18.51)
GRID = 24
c = np.linspace(-180 + 180 / GRID, 180 - 180 / GRID, GRID)
P, S = np.meshgrid(c, c)
grid_inits_phi = P.ravel()
grid_inits_psi = S.ravel()

ax1.scatter(grid_inits_phi, grid_inits_psi, color='gray', alpha=0.3, s=10, label='576 Init Grid')
ax1.scatter(-50.07, 18.51, color='blue', s=120, marker='X', zorder=5, label='Monotone Fixed Point ($N=1$)')
for i in range(0, len(grid_inits_phi), 16):
    ax1.annotate('', xy=(-50.07, 18.51), xytext=(grid_inits_phi[i], grid_inits_psi[i]),
                 arrowprops=dict(arrowstyle="->", color='blue', alpha=0.2, lw=0.8))

ax1.set_xlabel(r'$\phi$ [degrees]')
ax1.set_ylabel(r'$\psi$ [degrees]')
ax1.set_xlim([-180, 180])
ax1.set_ylim([-180, 180])
ax1.set_title('Monotone DEQ: Banach Contraction ($N=1$)\n[PENDING: auditor check]')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.3)

# Panel 2: EquiPhase DEQ (Multistable Branching)
ax2.scatter(grid_inits_phi, grid_inits_psi, color='gray', alpha=0.3, s=10, label='576 Init Grid')
ax2.scatter(-72.82, 153.38, color='red', s=100, label=r'$\beta$ Attractor', zorder=5)
ax2.scatter(-77.55, -14.97, color='green', s=100, label=r'$\alpha_R$ Attractor', zorder=5)
ax2.scatter(52.31, 32.76, color='purple', s=100, label=r'$\alpha_L$ Rare State', zorder=5)

ax2.set_xlabel(r'$\phi$ [degrees]')
ax2.set_ylabel(r'$\psi$ [degrees]')
ax2.set_xlim([-180, 180])
ax2.set_ylim([-180, 180])
ax2.set_title('EquiPhase DEQ: Multistable Branching\n[PENDING: auditor check]')
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PNG)
plt.close()

h = hashlib.sha256(open(OUT_PNG, 'rb').read()).hexdigest().upper()
print(f"Generated {OUT_PNG} | SHA256={h}")
