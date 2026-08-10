# EquiPhase Pre-Registration Amendment E-A & E-B

- **Author**: Sanghoon Huh (허상훈)
- **Date**: 2026-08-11
- **Status**: LOCKED & SEALED

---

## 1. Amendment E-A: Helmholtz-Projected Baseline Discrepancy Protocol

To evaluate the hypothesis that "structure buys a consistent free-energy readout," we establish a protocol to verify whether an unconstrained (vanilla) score matching model yields projection-dependent free-energy values ($\Delta F$) due to its non-zero curl.

### Protocol Details:
1. Train a Vanilla Score model on the MD backbone-dihedral dataset (seed 7777, noise $\sigma = 0.15$, 3000 steps, batch 4096).
2. Compute the 2D vector field $F = -\text{score}(q)$ on a $256 \times 256$ grid.
3. Apply two different numerical Poisson solvers to project the non-conservative field to a scalar potential $\tilde{V}$:
   - **Fourier-Spectral Solver**:
     $$\tilde{V}_{\text{spectral}}(k) = \frac{-i k_\phi F_\phi(k) - i k_\psi F_\psi(k)}{k_\phi^2 + k_\psi^2}$$
   - **Finite-Difference Solver**: Solve $\nabla^2 \tilde{V} = \nabla \cdot (-F)$ using a standard five-point Laplace stencil on the grid.
4. Extract the potential values at the coordinates of the $\beta$ and $\alpha_R$ attractors found in the EquiPhase model.
5. Compute the free energy difference $\Delta F(\alpha_R - \beta) = \tilde{V}(\alpha_R) - \tilde{V}(\beta)$ under both projections.
6. The test is considered successful if $|\Delta F_{\text{spectral}} - \Delta F_{\text{FD}}| > 0.1\,k_B T$, proving that the unconstrained baseline yields physically inconsistent, discretization-dependent free-energy readouts.

---

## 2. Amendment E-B: η/σ Robustness Sweep and Depth-Margin Lock

To evaluate the stability of the learned potential and solver convergence under different dynamical parameters, we establish a grid sweep over noise levels $\sigma$ and damping coefficients $\eta$.

### Parameter Sweep Grid:
- Noise levels: $\sigma \in \{0.05, 0.10, 0.15, 0.25\}$
- Damping coefficients: $\eta \in \{0.05, 0.10, 0.20, 0.50, 0.90\}$
- Total configurations: 20

### Evaluation Protocol:
For each configuration $(\sigma, \eta)$, we train the potential network `VNet` and integrate 576 initial grid points using conformal symplectic Euler dynamics for 2000 steps ($dt = 0.05$). We report:
1. **$N_{\text{basins}}$**: Number of stable attractors resolved after clustering.
2. **R1, R2, R3 Gates**: Pass/fail status of location, macrostate identity, and free-energy ordering.
3. **Convergence Rate**: Fraction of trajectories that do not diverge ($\|q\| < 10^4$).

### Attractor Classification Decision Rule (Depth-Margin Lock):
To eliminate post-hoc bias in determining whether an attractor is a physical state or a numerical artifact, we lock the following threshold:
- **Depth-Margin Threshold**: Let $V_{\text{min}}$ be the minimum potential depth among all resolved attractors. An attractor at coordinate $q^*$ is classified as a **numerical artifact** if:
  $$V(q^*) - V_{\text{min}} > 10.0\,k_B T$$
- Physical macrostates ($\beta, \alpha_R, \alpha_L$) must satisfy $V(q^*) - V_{\text{min}} \le 10.0\,k_B T$. Any state violating this bound is excluded from the physical attractor basin count $N_{\text{basins}}$ and reported as an artifact.
