# Paper 2 Official Preregistration v3: 32D Anisotropic Double-Well Conformal Symplectic DEQ (2D Active Subspace)

## 1. Executive Summary & Preregistration Protocol Compliance

This document constitutes the official, mathematically locked preregistration v3 for Paper 2 (*Conformal Symplectic Equilibrium Learning in Multistable Neural Dynamical Systems*).

### Protocol §4.3 Preregistration Locking Declaration:
- **Lock Timestamp**: 2026-08-08T08:35:00+09:00
- **Verification Rule**: This preregistration document is committed to Git **prior to** launching the confirmation training run under a **new random seed (`seed = 7777`)**. All prior runs are designated as exploratory.

---

## 2. Revision of FREEZE Condition (1) & Architecture Specifications

### FREEZE Condition (1) Amendment (Date: 2026-08-08)
- **Previous Specification**: Momentum-gated damping $\gamma(p) = \gamma_0 \cdot \sigma(k(0.10 - \|p\|))$.
- **Revision Reason**: Dynamical deadlock proof — trajectories starting outside the gate ($\|p_0\| \sim 5.6$) experience zero damping ($\gamma = 0$), preventing energy dissipation and locking trajectories out of the equilibrium region forever.
- **Amended Specification**: **Constant Damped Symplectic Euler (Semi-Implicit Euler with linear damping $\eta = 0.20$)**.

### State Space & Dimensional Definitions:
1. **State Vector**: $z = (q, p) \in \mathbb{R}^{64}$ ($q \in \mathbb{R}^{32}, p \in \mathbb{R}^{32}$).
2. **Active Subspace**: Multistable dynamics occur in 2D subspace $(q_1, q_2)$ (double-well along $q_1$, saddle along $q_2$). The remaining 30 dimensions $q_3 \dots q_{32}$ represent bound harmonic modes ($a_i = -0.5$).
3. **Official Designation**: **32D Anisotropic Double-Well DEQ (2D Active Subspace)**.

### Matrix A & Hessian Correctness:
- Potential: $V(q) = \frac{1}{4}\|q\|^4 - \frac{1}{2} q^\top A q$, where $A = \operatorname{diag}(a_1, a_2, a_3, \dots, a_{32})$ with $a_1 = \alpha \in [0.8, 1.2]$ (input conditioning), $a_2 = 0.3$, $a_{3\dots 32} = -0.5$. $A$ is strictly NOT a trainable parameter.
- Exact Hessian Formula:
  $$\nabla^2 V(q) = \|q\|^2 I + 2 q q^\top - A$$
- Curvature along active axes:
  - At minimum $q^* = \pm \sqrt{a_1} e_1$: $v_1 = e_1$ direction curvature is $\|q\|^2 + 2 q_1^2 - a_1 = a_1 + 2 a_1 - a_1 = 2 a_1$ (for $\alpha \in [0.8, 1.2]$, curvature $2 a_1 \in [1.6, 2.4]$).
  - At saddle $q^*_{\text{saddle}} = \pm \sqrt{a_2} e_2$: $v_2 = e_2$ direction curvature is $\|q\|^2 + 2 q_2^2 - a_2 = a_2 + 2 a_2 - a_2 = 2 a_2 = 0.6$.

---

## 3. Sign-Paired Task (c) DEQ Supervised Learning Specification

$$\mathcal{L}_{\text{DEQ}}(\theta) = \frac{1}{B} \sum_{i=1}^B \left\| \text{solve}(f_\theta; x^{(i)}, z_0^{(i)}) - \operatorname{sign}(z_{0, q1}^{(i)}) \sqrt{\alpha^{(i)}} e_1 \right\|_2^2 + 10 \cdot \| z^* - f_\theta(z^*; x^{(i)}) \|_2^2$$

- **Backpropagation Mode**: Locked exclusively to **DEQ Implicit Function Theorem (IFT) Autograd Pass**.
- **Batch Balance**: Exactly 50% positive ($+q_0$) and 50% negative ($-q_0$) initializations per batch.
- **Fixed Loss Guarantee**: The loss formulation and sign-paired target definitions are locked and shall NOT be modified post-hoc under any circumstances.

---

## 4. Theoretical Derivation of Verification Gates & Thresholds

1. **G3' Trajectory Residual ($\|z_{601} - z_{600}\|_2 < 1.0 \times 10^{-6}$)**:
   - Solver steps extended to 600 steps to guarantee contraction under damping $\eta = 0.20$.
2. **G4a' Basin Multistability ($N_{\text{stable\_basins}} \ge 2$)**:
   - Requires presence of at least 2 stable attractors ($\rho(J_f) < 1.0$).
3. **G4b' Dominant Basin Share ($\text{dominant\_share} \ge 0.90$)**:
   - Fraction of trajectories converging to primary double-well attractors $\pm \sqrt{\alpha} e_1$ must be $\ge 90\%$.
4. **G5' Neural Minimum Displacement ($\|\Delta q^*\| \le \frac{1.0 \times 10^{-2}}{2 \alpha}$ or worst-case $\le 6.25 \times 10^{-3}$ for $\alpha = 0.8$)**:
   - Derived from minimum curvature $2 a_1 \ge 1.6$. For gradient approximation error $\|\nabla \epsilon\| \le 1.0 \times 10^{-2}$, displacement $\|\Delta q^*\| \le \frac{1.0 \times 10^{-2}}{1.6} = 6.25 \times 10^{-3}$.
5. **G6' Neural Saddle Displacement ($\|\Delta q^*_{\text{saddle}}\| \le 1.67 \times 10^{-2}$)**:
   - Derived from saddle curvature $2 a_2 = 0.6$. For gradient approximation error $\|\nabla \epsilon\| \le 1.0 \times 10^{-2}$, displacement $\|\Delta q^*_{\text{saddle}}\| \le \frac{1.0 \times 10^{-2}}{0.6} = 1.67 \times 10^{-2}$.
6. **G7' Energy Barrier Match ($\|V(\text{saddle}) - V(\text{min})\| = 0.2275 \pm 0.0100$)**:
   - Analytical energy difference $V(\text{saddle}) - V(\text{min}) = (-0.0225) - (-0.2500) = 0.2275$. Tolerance $\pm 0.0100$ accounts for path integral of gradient approximation error $\|\nabla \epsilon\| \le 10^{-2}$ over path length $\approx 1.0$.

---

## 5. Preregistered Verification Suite v3

| Gate ID | Physical/Numerical Requirement | Preregistered Criterion | Audit Classification |
|---|---|---|:---:|
| **G1** | Force Field Anti-Symmetry | $\frac{\|J_F - J_F^\top\|_F}{\|J_F\|_F} < 10^{-5}$ | **[Architectural Invariant]** |
| **G2** | Conformal Symplectic Conservation | $R < 10^{-6}$ & $c = 0.8000000$ | **[Architectural Invariant]** |
| **G3'** | Fixed-Point Trajectory Residual | $\|z_{601} - z_{600}\|_2 < 1.0 \times 10^{-6}$ | **[Task Metric]** |
| **G4a'** | Attractor Basin Multistability | $N_{\text{stable\_basins}} \ge 2$ with $\rho(J_f) < 1.0$ | **[Task Metric]** |
| **G4b'** | Dominant Well Concentration | $\text{dominant\_share} \ge 0.90$ | **[Task Metric]** |
| **G5'** | Neural Minimum Displacement | $\|\Delta q^*\| \le 6.25 \times 10^{-3}$ ($\alpha = 0.8$ worst-case) | **[Task Metric]** |
| **G6'** | Neural Saddle Displacement | $\|\Delta q^*_{\text{saddle}}\| \le 1.67 \times 10^{-2}$ | **[Task Metric]** |
| **G7'** | Energy Barrier Match | $\|V(\text{saddle}) - V(\text{min})\| = 0.2275 \pm 0.0100$ | **[Task Metric]** |

---

## 6. Three-Way Baseline Comparison Expectation Declaration

- **Vanilla DEQ (Baseline 1)**: Vanilla DEQ (unconstrained vector field) does not satisfy conformal symplecticity because $J_F$ is non-symmetric. The magnitude of $R$ is not pre-predicted and will be measured and reported post-hoc. Trajectories are expected to diverge or fail to converge based on previous observations, but exact $N_{\text{basins}}$ will be reported strictly as measured.
- **Monotone DEQ (Baseline 2)**: Monotone DEQ is mathematically proven by the Contraction Mapping Theorem to have a unique fixed point ($N_{\text{basins}} = 1$), serving as a structural baseline proving that monotone parameterization destroys multistability.
- **EquiPhase DEQ (Ours)**: Expected to preserve discrete conformal symplecticity ($c = 0.8000000, R < 10^{-6}$) and multistable attractor basins ($\text{dominant\_share} \ge 0.90$) post-training under new random seeds.

---

## 7. Unresolved Technical Discrepancies Recorded for Tracking

1. **G4 vs G6 Saddle Point Count**: Analytical potential has 2 saddle points $\pm \sqrt{0.3} e_2 = \pm 0.547723 e_2$; empirical trajectory distribution will be tracked.
2. **Explicit Euler $c$ Discrepancy**: Explicit Euler update sequence with $\Delta t = 0.1$ gave measured $c = 0.8087$ ($\text{tr}(J_F) \approx -27.8$). Damped Symplectic Euler (Semi-Implicit) guarantees exact $c = 1-\eta = 0.8000000$.
