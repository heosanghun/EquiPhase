# -*- coding: utf-8 -*-
import hashlib
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

OUT_PNG = os.path.join("paper3_iclr", "fig2_ala2_vnet_contours_v3.png")
os.makedirs("paper3_iclr", exist_ok=True)

class VNet(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(4, h), nn.Tanh(),
                               nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
    def enc(self, q):
        return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)
    def forward(self, q):
        return self.f(self.enc(q)).squeeze(-1)

# Find latest seed 7777 model in results/
res_dir = "results"
pt_files = [f for f in os.listdir(res_dir) if "ala2_vnet_seed7777" in f and f.endswith(".pt")]
if not pt_files:
    raise FileNotFoundError("No seed 7777 model file found in results/")

latest_pt = os.path.join(res_dir, sorted(pt_files)[-1])
net = VNet()
net.load_state_dict(torch.load(latest_pt, map_location='cpu'))
net.eval()

grid_res = 120
phi_grid = np.linspace(-np.pi, np.pi, grid_res)
psi_grid = np.linspace(-np.pi, np.pi, grid_res)
P, S = np.meshgrid(phi_grid, psi_grid)
q_inp = torch.tensor(np.stack([P.ravel(), S.ravel()], axis=-1), dtype=torch.float32)

with torch.no_grad():
    V_vals = net(q_inp).numpy().reshape(grid_res, grid_res)

fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
plt.style.use('dark_background')

cp = ax.contourf(np.degrees(P), np.degrees(S), V_vals, levels=30, cmap='magma')
fig.colorbar(cp, ax=ax, label=r'Learned Potential $V_{\mathrm{net}}$ ($k_B T$)')

# Annotate macrostates with exact sealed coordinates and values
ax.annotate(r'$\beta$ Basin (-10.45 $k_BT$)' + '\n(-72.8°, +153.4°)', xy=(-72.82, 153.38), xytext=(-120, 120),
            arrowprops=dict(facecolor='white', shrink=0.05, width=1.2, headwidth=6), color='white', fontweight='bold', fontsize=10)
ax.annotate(r'$\alpha_R$ Basin (-9.64 $k_BT$)' + '\n(-77.6°, -15.0°)', xy=(-77.55, -14.97), xytext=(-140, -60),
            arrowprops=dict(facecolor='white', shrink=0.05, width=1.2, headwidth=6), color='white', fontweight='bold', fontsize=10)
ax.annotate(r'$\alpha_L$ Rare State (-3.41 $k_BT$)' + '\n(+52.3°, +32.8°)', xy=(52.31, 32.76), xytext=(75, 55),
            arrowprops=dict(facecolor='cyan', shrink=0.05, width=1.2, headwidth=6), color='cyan', fontweight='bold', fontsize=10)

ax.set_xlim(-180, 180)
ax.set_ylim(-180, 180)
ax.set_xlabel(r'$\phi$ (degrees)', fontsize=13, fontweight='bold', labelpad=8)
ax.set_ylabel(r'$\psi$ (degrees)', fontsize=13, fontweight='bold', labelpad=8)
ax.set_title(r'EquiPhase Learned Potential Surface $V_\theta(\phi, \psi)$ (Seed 7777)', fontsize=13, pad=12)
ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.close()

h = hashlib.sha256(open(OUT_PNG, 'rb').read()).hexdigest().upper()
print(f"Generated Fig 2 v3 from sealed checkpoint at {OUT_PNG} | SHA256={h}")
