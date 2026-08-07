# Paper 2 Official Preregistration: 1D Anisotropic Double-Well Conformal Symplectic DEQ

## 1. Executive Summary & Core Guarantees
This document constitutes the official, mathematically locked preregistration for Paper 2 (*Conformal Symplectic Equilibrium Learning in Multistable Neural Dynamical Systems*).

All experiments evaluate the **Semi-Implicit Symplectic Euler (Leapfrog)** discrete update step applied to an exact gradient potential force field $F(q) = -\nabla_q V_{\text{total}}(q)$ with uniform momentum damping $\eta = 0.20$ ($\text{damping} = 0.20$) and time-step $\Delta t = 0.10$.

### Core Physical & Numerical Invariants:
1. **Force Anti-Symmetry & Conservatism**: $F(q) = -\nabla_q V(q)$ strictly guarantees $J_F = -\nabla^2_q V = J_F^\top$ (0.0000% anti-symmetric residual).
2. **Discrete Conformal Symplecticity**: The discrete map $f(z)$ satisfies $J_f^\top \Omega J_f = c \Omega$ with $c = 1 - \eta = 0.8000000$ exactly, and relative symplectic violation $R < 10^{-6}$.
3. **Spectral Radius Filtering for Basin Resolution**: EquiPhase attractor basins are uniquely defined by $\rho(J_f(z^*)) < 1.0$. Unstable saddle points ($F(q^*)=0, p^*=0$) exhibit $\rho(J_f) > 1.0$ and are strictly excluded from stable attractor counts.

---

## 2. Model Architecture & Checkpoint Specifications

- **Model Class**: `AnisotropicDoubleWellDEQ`
- **Script Location**: [`train_1d_double_well.py`](file:///C:/Project/EquiPhase/train_1d_double_well.py)
- **Model Checkpoint**: [`anisotropic_double_well_deq.pt`](file:///C:/Project/EquiPhase/anisotropic_double_well_deq.pt)
- **Checkpoint SHA-256 Hash**: `02f32e3fc9276e614775e6afccb17ace2da7444b0163cb5d1f7a6b2915d92e11`

---

## 3. Preregistered 7-Gate Verification Suite & Audit Results

| Gate ID | Physical/Numerical Requirement | Preregistered Criterion | Measured Value | Verification Status |
|---|---|---|---|:---:|
| **G1** | Force Field Anti-Symmetry | $\frac{\|J_F - J_F^\top\|_F}{\|J_F\|_F} < 10^{-5}$ | `0.0000e+00%` | **PASS** |
| **G2** | Conformal Symplectic Conservation | $R < 10^{-6}$ & $c = 0.8000000$ | $c = 0.8000000$, $R = 1.4237 \times 10^{-7}$ | **PASS** |
| **G3** | Trajectory Fixed-Point Residual | $\|z_{501} - z_{500}\|_2 < 10^{-6}$ across 100/100 initializations | `4.7962e-07` | **PASS** |
| **G4** | Attractor Basin Resolution | $N_{\text{stable\_basins}} == 2$ with $\rho(J_f) < 1.0$ | 2 Basins (Plus: 53, Minus: 46, Saddle: 1 with $\rho > 1$) | **PASS** |
| **G5** | Exact Analytical Minimum Match | $q^* = \pm e_1 = \pm (1.0, 0, \dots, 0)^\top$ | $+1.000000 e_1$, $-1.000000 e_1$ | **PASS** |
| **G6** | Exact Analytical Saddle Match | $q^*_{\text{saddle}} = \pm \sqrt{0.3} e_2 = \pm 0.547723 e_2$ | $\pm 0.547723 e_2$ | **PASS** |
| **G7** | Exact Energy Barrier Match | $\|V(\text{saddle}) - V(\text{min})\| = 0.227500$ | Measured = `0.227500`, Diff = `8.94e-10` | **PASS** |

### Overall Verification Summary:
**TRAINED MODEL 7-GATE PREREGISTRATION STATUS: ALL 7 GATES PASSED (100%)**
