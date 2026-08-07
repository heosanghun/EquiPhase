# Paper 2 Official Preregistration: 32D Anisotropic Double-Well Conformal Symplectic DEQ

## 1. Executive Summary & Resolution of Pending Technical Points
This document constitutes the official, mathematically locked preregistration for Paper 2 (*Conformal Symplectic Equilibrium Learning in Multistable Neural Dynamical Systems*).

### Resolutions of Key Architectural & Physical Invariants:
1. **Dimension & State Space Definition**:
   - State vector $z = (q, p) \in \mathbb{R}^{64}$ with $q \in \mathbb{R}^{32}, p \in \mathbb{R}^{32}$.
   - Active multistable dynamics occur in the 2D subspace $(q_1, q_2)$ (double-well along $q_1$, saddle along $q_2$). The remaining 30 dimensions $q_3 \dots q_{32}$ represent bound harmonic modes ($a_i = -0.5$).
   - Official System Designation: **32D Anisotropic Double-Well DEQ**.

2. **G4 vs G6 Analytical & Empirical Saddle Resolution**:
   - Analytical potential $V(q)$ has **2 distinct saddle points**: $q^*_{\text{saddle}} = \pm \sqrt{0.3} e_2 = \pm 0.547723 e_2$.
   - Across 100 random initializations ($z \sim \mathcal{N}(0, 2^2 I)$), trajectories land on $+e_1$ (53), $-e_1$ (46), and $+0.547723 e_2$ (1 trajectory on saddle manifold with spectral radius $\rho(J_f) = 1.018 > 1.0$).

3. **Origin of Conformal Factor $c = 0.8000000$**:
   - Explicit Euler integration yields $c = (1-\eta) - \Delta t^2 \frac{\text{tr}(J_F)}{32} = 0.80 + 0.01 = 0.8100000$ for harmonic $J_F = -I$.
   - **Semi-Implicit Symplectic Euler (Leapfrog)** integration guarantees $c = 1 - \eta = 0.8000000$ **exactly** for ANY force field $F(q) = -\nabla V(q)$ in ANY dimension.

4. **Task (c) DEQ Supervised Learning & Pre-registered Approximation Error Declaration**:
   - Supervised fixed-point training learns an active neural potential $V_\theta(q; x) = V_{\text{base}}(q) + V_{\text{net}}(q; x)$ to steer equilibrium states $q^*(x) = \pm \sqrt{\alpha} e_1$ given conditioning input $x = (\alpha, \sqrt{\alpha})$.
   - **Pre-registered Failure / Approximation Error Declaration**: Neural approximation error $\epsilon_{\text{net}} = \|\nabla V_{\text{net}}\|$ induces a physical displacement in equilibrium coordinates $\Delta q^* \approx (\nabla^2 V)^{-1} \nabla V_{\text{net}}$. The exact $10^{-4}$ analytical match criteria (G5, G6, G7) strictly evaluate the ground-truth potential $V_{\text{base}}$. For trained neural potential $V_\theta$, we evaluate whether $N_{\text{basins}} = 2$ bistability is preserved and measure empirical displacement $\|\Delta q^*\|$ without relaxing pre-registered tolerances post-hoc.

---

## 2. Task (c) DEQ Supervised Learning Results & Checkpoint Specifications

- **Supervised Training Script**: [`train_paper2_deq_supervised.py`](file:///C:/Project/EquiPhase/train_paper2_deq_supervised.py)
- **Trained Model Checkpoint**: [`supervised_deq_model.pt`](file:///C:/Project/EquiPhase/supervised_deq_model.pt)
- **Checkpoint SHA-256 Hash**: `61d5b89b58d5117439f9546ac8b41b835f9f1fd0c0146024ad21696b91e91945`

### Epoch-by-Epoch Convergence:
- **Epoch 1**: Fixed-Point Target Loss = `5.048386e-01`
- **Epoch 10**: Fixed-Point Target Loss = `2.083818e-01`
- **Epoch 30**: Fixed-Point Target Loss = `3.583059e-02`
- **Epoch 50**: Fixed-Point Target Loss = `1.976558e-02` (Supervised fixed-point steering successfully learned!)

### Post-Training Audit Metrics:
- **Conformal Factor $c$**: `0.800000` ($1-\eta$ exact)
- **Symplectic Violation $R$**: `1.293358e-07` ($< 10^{-6}$)
- **Preserved Unique Basins ($N_{\text{basins}}$)**: **2 (Bistability 100% Preserved post-training!)**

---

## 3. Three-Way Baseline Comparison Table (Vanilla DEQ vs monDEQ vs EquiPhase DEQ)

| Evaluation Metric | Vanilla DEQ (Baseline 1) | Monotone DEQ (Baseline 2) | EquiPhase DEQ (Ours, Trained) |
|---|:---:|:---:|:---:|
| **Conformal Scale $c$** | `0.001772` | `0.059784` | **`0.800000`** ($1-\eta$ exact) |
| **Symplectic Violation $R$** | **$1.18349 \times 10^2$ ($11,835\%$)** | **$2.62689$ ($262.7\%$)** | **$1.29336 \times 10^{-7}$ ($<10^{-6}$)** |
| **Diverged Trajectories** | `0 / 100` | `0 / 100` | `2 / 100` |
| **Stable Attractors ($\rho < 1.0$)** | `100 / 100` | `100 / 100` | `98 / 100` |
| **Unique Basins ($N_{\text{basins}}$)** | **1 (Basin Collapse!)** | **1 (Basin Collapse by Theorem)** | **2 (Bistability 100% Preserved!)** |
