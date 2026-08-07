# Paper 2 Official Preregistration v3: 32D Anisotropic Double-Well Conformal Symplectic DEQ (2D Active Subspace)

## 1. Executive Summary & Resolution of Pending Technical Points

This document constitutes the official, mathematically locked preregistration v3 for Paper 2 (*Conformal Symplectic Equilibrium Learning in Multistable Neural Dynamical Systems*).

### Resolution of Architectural & Physical Invariants:

1. **State Space & Dimensional Definition**:
   - State vector $z = (q, p) \in \mathbb{R}^{64}$ ($q \in \mathbb{R}^{32}, p \in \mathbb{R}^{32}$).
   - Active multistable dynamics occur in the 2D subspace $(q_1, q_2)$ (double-well along $q_1$, saddle along $q_2$). The remaining 30 dimensions $q_3 \dots q_{32}$ represent bound harmonic modes ($a_i = -0.5$).
   - Official Designation: **32D Anisotropic Double-Well DEQ (2D Active Subspace)**.

2. **A Matrix Parameterization**:
   - Anisotropy parameter $a_1 = \alpha \in [0.8, 1.2]$ is passed dynamically via conditioning vector $x = (\alpha, \sqrt{\alpha})$.
   - Saddle anisotropy $a_2 = 0.3$ and harmonic coefficients $a_{3\dots 32} = -0.5$ are fixed physical constants.
   - $A$ is strictly NOT a trainable parameter matrix.

3. **Curvature Calculation Correctness**:
   - For potential $V(q) = \frac{1}{4}\|q\|^4 - \frac{1}{2} q^\top A q$, Hessian is $\nabla^2 V(q) = 3 q_1^2 e_1 e_1^\top + 3 q_2^2 e_2 e_2^\top - A$.
   - At minimum $q^* = \pm e_1$: $a_1 = 1.0 \implies \text{curvature} = 3(1.0)^2 - a_1 = 2 a_1 = 2.0$.
   - At saddle $q^*_{\text{saddle}} = \pm \sqrt{0.3} e_2$: $a_2 = 0.3 \implies \text{curvature} = 3(0.3) - a_2 = 2 a_2 = 0.6$.

---

## 2. Sign-Paired Task (c) DEQ Supervised Learning & Loss Formulation

To prevent basin collapse during supervised learning, targets are paired according to initial orientation:
$$\mathcal{L}_{\text{DEQ}}(\theta) = \frac{1}{B} \sum_{i=1}^B \left\| \text{solve}(f_\theta; x^{(i)}, z_0^{(i)}) - \operatorname{sign}(z_{0, q1}^{(i)}) \sqrt{\alpha^{(i)}} e_1 \right\|_2^2 + 10 \cdot \| z^* - f_\theta(z^*; x^{(i)}) \|_2^2$$

- **Backpropagation Mode**: Explicitly locked to **DEQ Implicit Function Theorem (IFT) Autograd Pass** (backward solver pass).
- **Batch Balance**: Exactly 50% positive ($+q_0$) and 50% negative ($-q_0$) initializations per batch.

### Pre-registered Failure & Approximation Error Declarations:
1. **Fixed Loss & Target Specification**: The sign-paired target formulation and target definitions above are locked and shall NOT be modified post-hoc under any circumstances.
2. **G3' Solver Trajectory Extension**: Solver iteration steps extended to 600 steps to guarantee trajectory residual $\|z_{601} - z_{600}\|_2 < 10^{-6}$ under damping $\eta = 0.20$.
3. **G5' Minimum Threshold ($5 \times 10^{-3}$)**: Derived from minimum curvature $2 a_1 = 2.0$. Neural gradient error $\|\nabla \epsilon\| \le 10^{-2}$ bounds displacement to $\|\Delta q^*\| \le \frac{10^{-2}}{2.0} = 5 \times 10^{-3}$.
4. **G6' Saddle Threshold ($8.3 \times 10^{-3}$)**: Derived from saddle curvature $2 a_2 = 0.6$ (1.67x softer than $v_1$), yielding threshold $\frac{5 \times 10^{-3}}{0.6} = 8.3 \times 10^{-3}$.
5. **G4b Dominant Basin Share ($\ge 0.90$)**: Spurious local minima from neural MLP $V_\theta$ are permitted ($N_{\text{basins}} \ge 2$), but the fraction of trajectories converging to the primary double wells $\pm e_1$ must satisfy $\text{dominant\_share} \ge 0.90$.

---

## 3. Preregistered Verification Suite v3 & Audit Classification

| Gate ID | Physical/Numerical Requirement | Preregistered Criterion | Audit Classification |
|---|---|---|:---:|
| **G1** | Force Field Anti-Symmetry | $\frac{\|J_F - J_F^\top\|_F}{\|J_F\|_F} < 10^{-5}$ | **[Architectural Invariant]** |
| **G2** | Conformal Symplectic Conservation | $R < 10^{-6}$ & $c = 0.8000000$ | **[Architectural Invariant]** |
| **G3'** | Fixed-Point Trajectory Residual | $\|z_{601} - z_{600}\|_2 < 10^{-6}$ | **[Task Metric]** |
| **G4a'** | Basin Multistability (No Collapse) | $N_{\text{stable\_basins}} \ge 2$ with $\rho(J_f) < 1.0$ | **[Task Metric]** |
| **G4b'** | Dominant Well Concentration | $\text{dominant\_share} \ge 0.90$ | **[Task Metric]** |
| **G5'** | Neural Minimum Displacement | $\|\Delta q^*\| \le 5 \times 10^{-3}$ | **[Task Metric]** |
| **G6'** | Neural Saddle Displacement | $\|\Delta q^*_{\text{saddle}}\| \le 8.3 \times 10^{-3}$ | **[Task Metric]** |
| **G7'** | Energy Barrier Match | $\|V(\text{saddle}) - V(\text{min})\| = 0.2275 \pm 0.01$ | **[Task Metric]** |

---

## 4. Three-Way Baseline Comparison Expectation Declaration

- **Vanilla DEQ (Baseline 1)**: Vanilla DEQ (unconstrained vector field) does not satisfy conformal symplecticity because $J_F$ is non-symmetric. The magnitude of $R$ is not pre-predicted and will be measured and reported post-hoc. Trajectories are expected to diverge or fail to converge based on previous observations, but exact $N_{\text{basins}}$ will be reported strictly as measured.
- **Monotone DEQ (Baseline 2)**: Monotone DEQ is mathematically proven by the Contraction Mapping Theorem to have a unique fixed point ($N_{\text{basins}} = 1$), serving as a structural baseline proving that monotone parameterization destroys multistability.
- **EquiPhase DEQ (Ours)**: Expected to preserve discrete conformal symplecticity ($c = 0.8000000, R < 10^{-6}$) and multistable attractor basins ($\text{dominant\_share} \ge 0.90$) post-training.

---

## 5. Unresolved Technical Discrepancies Recorded for Tracking

1. **G4 vs G6 Saddle Point Count**: Analytical potential has 2 saddle points $\pm \sqrt{0.3} e_2 = \pm 0.547723 e_2$; 100 random initializations landed 53 at $+e_1$, 46 at $-e_1$, 1 at $+\sqrt{0.3} e_2$.
2. **Origin of $c=0.81$ under Explicit Euler**: Explicit Euler update sequence with $\Delta t = 0.1$ gives $c = (1-\eta) - \Delta t^2 \text{tr}(J_F)/32 = 0.80 + 0.01 = 0.8100000$. Semi-Implicit Leapfrog yields exact $c = 1-\eta = 0.8000000$.
