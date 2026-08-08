# Step 12 Baseline Preregistration (frozen prior to execution)

**Date**: 2026-08-08  
**Sealed Script**: `claude_paper2_baselines_sealed.py` (SHA-256 `fd8f2b525530ec3de465ae1172e92aa4658369bcc204177d5155307e5ef9f74f`)  
**Training Seed**: 7777  
**Audit Inits**: generator seed 314159 (identical to EquiPhase sealed audit)  
**Audit Steps**: 600  
**Divergence Threshold**: $\|z\| > 10^4$

---

## Baseline Specifications

### Baseline 1 (Vanilla DEQ)
- **Architecture**: Unconstrained force field MLP $34 \rightarrow 64 \rightarrow 32$, same damped velocity Verlet integrator.
- **Expectations (frozen)**:
  - $J_F$ force anti-symmetry measured with no magnitude predicted (random-matrix reference $139.2\% \pm 3.2\%$ cited for context).
  - Conformal $c / R$ measured post-hoc with no magnitude predicted.
  - Trajectories may diverge or converge; $N_{\text{endpoint\_clusters}}$ reported strictly as measured.
  - Training NaN/Inf epochs are skipped, counted, and reported as outcomes.

### Baseline 2 (Monotone DEQ)
- **Architecture**: Contraction map $z' = \tanh(Wz + Ux + b)$ with $\|W\|_2 \le 0.9$ via spectral normalization.
- **Expectations (frozen)**:
  - $N_{\text{basins}} = 1$ is a STRUCTURAL consequence of the Banach fixed-point theorem.
  - The audit run illustrates, rather than discovers, this single-attractor collapse.

---

## Shared Protocol Notes
- No pass/fail thresholds apply to baselines; all metrics are comparative.
- **Deviation Note**: Training uses 100-step unrolled backpropagation, identical to the EquiPhase arm (disclosed deviation applies to all arms equally).
