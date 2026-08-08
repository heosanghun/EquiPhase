# -*- coding: utf-8 -*-
# =============================================================================
# train_ala2_equiphase_draft.py
# DRAFT TRAINING SCRIPT — EquiPhase DEQ × Alanine Dipeptide Real Data
# Specification: PREREG_ALA2_EQUIPHASE_v1.md Section 3
# Rules        : Draft only. NO EXECUTION allowed before sealed gate script.
# =============================================================================
import hashlib
import os
import sys
import numpy as np
import torch
import torch.nn as nn

DATA_PATH = os.path.join("data", "ala2", "alanine-dipeptide-3x250ns-backbone-dihedrals.npz")
ANCHOR_SHA256 = "F5AD30768A7CF3451B3061CB2ECB7F7D1DE8C13044534376D26A6653D4CD5717"
SEEDS = [7777, 1234, 2026, 31415, 65537]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def load_and_verify_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    digest = sha256_of(DATA_PATH)
    if digest != ANCHOR_SHA256:
        raise ValueError(f"Data hash mismatch: {digest} vs {ANCHOR_SHA256}")
    npz = np.load(DATA_PATH)
    parts = [np.asarray(npz[k]).astype(np.float64) for k in sorted(npz.files)]
    X = np.concatenate(parts, axis=0)
    return X


def encode_dihedrals(angles):
    """Encodes 2D dihedrals (phi, psi) into 4D trigonometric embedding [sin phi, cos phi, sin psi, cos psi]."""
    phi, psi = angles[..., 0], angles[..., 1]
    return torch.stack([torch.sin(phi), torch.cos(phi), torch.sin(psi), torch.cos(psi)], dim=-1)


class EquiPhasePotentialMLP(nn.Module):
    """Scalar neural potential V_net(q; theta) for Alanine Dipeptide 2D dihedrals."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, q):
        # q is 2D dihedrals (phi, psi) in radians
        emb = encode_dihedrals(q)
        return self.net(emb).squeeze(-1)


class EquiPhaseDEQStep(nn.Module):
    """Damped velocity Verlet step satisfying exact modified symplectic preservation identity."""
    def __init__(self, v_net, dt=0.05, gamma=0.1, m=1.0):
        super().__init__()
        self.v_net = v_net
        self.dt = dt
        self.gamma = gamma
        self.m = m
        self.eta = gamma * dt / m

    def forward(self, q_t, p_t):
        # Half-step position
        q_half = q_t + 0.5 * (self.dt / self.m) * p_t
        q_half_req = q_half.detach().requires_grad_(True)
        v_val = self.v_net(q_half_req).sum()
        grad_v = torch.autograd.grad(v_val, q_half_req, create_graph=True)[0]
        
        # Momentum update with physical damping
        p_next = (1.0 - self.eta) * p_t - self.dt * grad_v
        
        # Full-step position
        q_next = q_half + 0.5 * (self.dt / self.m) * p_next
        # Wrap position into [-pi, pi) periodic domain
        q_next_wrapped = torch.remainder(q_next + np.pi, 2 * np.pi) - np.pi
        return q_next_wrapped, p_next


class MonotoneDEQStep(nn.Module):
    """Monotone DEQ control architecture enforcing ||W||_2 <= 0.9 contraction constraint."""
    def __init__(self, state_dim=4, hidden_dim=64, spectral_bound=0.9):
        super().__init__()
        self.fc = nn.utils.spectral_norm(nn.Linear(state_dim, hidden_dim))
        self.out = nn.utils.spectral_norm(nn.Linear(hidden_dim, state_dim))
        self.bound = spectral_bound

    def forward(self, z):
        h = torch.relu(self.fc(z))
        return self.bound * torch.tanh(self.out(h))


class VanillaDEQStep(nn.Module):
    """Unconstrained Vanilla DEQ control architecture."""
    def __init__(self, state_dim=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, z):
        return self.net(z)


if __name__ == "__main__":
    print("train_ala2_equiphase_draft.py — DRAFT ONLY (NO EXECUTION PERMITTED BEFORE SEALED GATES)")
    print("Checking dataset integrity...")
    data = load_and_verify_data()
    print(f"Dataset verified successfully: {data.shape} frames loaded.")
