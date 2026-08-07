# MS-DEQ + LLM Dual-Axis Agent Architecture: Dynamical Attractors, Operational Multi-Stability, and Measurement Metrics

**Author**: Sanghoon Huh (허상훈)  
**Date**: August 2026  
**Document Version**: 2.1 (Refined Formal Specification)

---

## 1. Executive Summary and Foundational Framework

High-stakes agentic systems require two distinct computational modes:
1. **Explicit Sequential Reasoning (Token Axis)**: Autoregressive token generation (e.g., Chain-of-Thought, tool call sequences).
2. **Implicit Deep Equilibrium Compute (Latent Axis)**: Iterative fixed-point solving over continuous representation spaces, enabling input-adaptive test-time compute depth.

In **Multi-Stable Deep Equilibrium Models (MS-DEQ)**, fixed-point iteration $z^{(k+1)} = f_\theta(z^{(k)}; x_t)$ operates over a phase space containing multiple attraction basins. **Multi-stability does not represent sequential reasoning; rather, it formalizes the coexistence of candidate hypothesis basins.** 

At any decision epoch $t$, the operational multi-stability count $N(x_t)$ represents the number of distinct candidate action pathways resulting from a finite set of initial candidate proposals. Operational constraints act as dynamical perturbations $u_t$ that induce saddle-node bifurcations, eliminating non-compliant attraction basins and collapsing the state into a single compliant action fixed point.

---

## 2. Mathematical Formalism (Pure Dynamical Systems Approach)

```
                   +-------------------------------------------------------+
                   |                 TOKEN AXIS (Sequential)               |
                   |   Autoregressive Reasoning: CoT / Tool Calling        |
                   +-------------------------------------------------------+
                                               |
                                     Context Vector x_t
                                               v
+--------------------------------------------------------------------------------------------------+
|                                    LATENT AXIS (Equilibrium)                                     |
|                                                                                                  |
|   Discrete Phase Space Mapping:  z^(k+1) = f_theta(z^(k); x_t)                                  |
|                                                                                                  |
|   Asymptotically Stable Fixed Points (Attractors):                                               |
|       z^*_1, z^*_2, ..., z^*_N  s.t.  z^*_k = f_theta(z^*_k; x_t) ,  rho(J_f(z^*_k)) < 1         |
|                                                                                                  |
|   Operational Constraint Perturbation:  u_t (Safety Policy / Guardrails)                         |
|       z^(k+1) = f_theta(z^(k); x_t, u_t)                                                         |
|                                                                                                  |
|   Saddle-Node Bifurcation -> Basin Collapse to Unique Compliant Attractor z*_(collapsed)          |
+--------------------------------------------------------------------------------------------------+
                                               |
                                    Collapsed Attractor z*
                                               v
                   +-------------------------------------------------------+
                   |                 TOKEN AXIS Output                     |
                   |   Next Action / Token Distribution P(y_t | y_<t, z*)  |
                   +-------------------------------------------------------+
```

### 2.1 Mathematical Definitions

#### Definition 1 (Token Axis Dynamics)
Let $y_{<t} = (y_1, y_2, \dots, y_{t-1})$ denote the sequence of emitted tokens and tool calls up to step $t$. The token axis generates explicit reasoning steps autoregressively:
$$P(y_t \mid y_{<t}, z_t^*) = \text{Softmax}\left( W_{\text{lm}} \cdot \text{TransformerLayer}(y_{<t}, z_t^*) \right)$$

#### Definition 2 (Attractor Fixed Points & Dynamical Stability)
At decision step $t$, context vector $x_t = \text{Embed}(y_{<t})$ conditions mapping $f_\theta(\cdot ; x_t): \mathbb{R}^d \to \mathbb{R}^d$. A point $z^* \in \mathbb{R}^d$ is an **asymptotically stable fixed point (attractor)** if and only if:
1. **Fixed Point Condition**: $z^* = f_\theta(z^*; x_t)$
2. **Local Asymptotic Stability**: Spectral radius $\rho\left( J_f(z^*) \right) < 1$, where $J_f(z^*) = \left. \frac{\partial f_\theta(z; x_t)}{\partial z} \right|_{z = z^*}$.
3. **Implicit Function Theorem Invertibility**: $\det(I - J_f(z^*)) \neq 0$.

#### Definition 3 (Basin of Attraction)
For an attractor $z_k^*$, its basin of attraction $\mathcal{B}(z_k^*)$ is defined as:
$$\mathcal{B}(z_k^*) \triangleq \left\{ z^{(0)} \in \mathbb{R}^d \;\middle|\; \lim_{m \to \infty} f_\theta^{\circ m}(z^{(0)}; x_t) = z_k^* \right\}$$

---

## 3. Operational Definitions & Metric Specifications

### 3.1 Operational Candidate Multi-Stability ($N_{\text{basins}}$)
Due to the curse of dimensionality ($d \approx 3000$ in modern LLMs), global basin volume sampling is intractable. We operationally define $N_{\text{basins}}$ relative to a finite set of $K$ candidate strategy initializations $\mathcal{Z}_0 = \{ z^{(0)}_1, z^{(0)}_2, \dots, z^{(0)}_K \}$ (generated via distinct prompt/tool candidate prefixes):

$$N_{\text{basins}}(x_t; \mathcal{Z}_0) \triangleq \left| \text{Cluster}_{\delta}\left( \left\{ \text{Solve}\left(f_\theta(\cdot; x_t), z^{(0)}_i\right) \;\middle|\; z^{(0)}_i \in \mathcal{Z}_0, \, \rho(J_f) < 1 \right\} \right) \right|$$

### 3.2 Attractor Sharpness & Hypothesis Confidence
- **Attractor Sharpness / Convergence Velocity ($V_{\text{conv}}$)**:
  $$V_{\text{conv}}(z_k^*) \triangleq -\ln \rho\left( J_f(z_k^*) \right)$$
  Higher $V_{\text{conv}}$ quantifies local attractor sharpness and numerical solver convergence speed.
- **Hypothesis Confidence / Basin Volume Share ($\text{Share}_k$)**:
  $$\text{Share}_k(\mathcal{Z}_0) \triangleq \frac{1}{K} \left| \left\{ i \in \{1, \dots, K\} \;\middle|\; \text{Solve}(f_\theta(\cdot; x_t), z^{(0)}_i) \to z_k^* \right\} \right|$$
  $\text{Share}_k$ measures the empirical probability weight of candidate attractor $z_k^*$ over proposal set $\mathcal{Z}_0$.

### 3.3 Critical Saddle-Node Bifurcation Threshold ($u^*$)
Operational constraint $u_t$ (safety policy perturbation) modifies $f_\theta(z; x_t, u_t)$. A saddle-node bifurcation occurs when a real eigenvalue of the Jacobian passes $+1$:
$$\lambda_{\max}\left( J_f(z_k^*; u) \right) = +1$$
The critical collapse threshold for eliminating non-compliant attractor $k$ is:
$$u_k^* \triangleq \inf \left\{ \|u\| \;\middle|\; \exists \lambda_i\left( J_f(z_k^*; u) \right) = +1 \right\}$$
At $u = u_k^*$, $I - J_f(z_k^*; u)$ becomes singular, destroying attractor $z_k^*$ and collapsing the latent phase space into the compliant basin.

---

## 4. Integration Architecture: Gemma 4 + MS-DEQ

1. **Token Axis (Gemma 4)**: Handles token sequence parsing, tool formatting, and explicit Chain-of-Thought generation.
2. **Latent Axis (MS-DEQ)**: Executes solver iterations over intermediate transformer state spaces to reach steady-state attractors $z^*$.
3. **Safety Control Perturbation**: Enforces operational constraints $u_t$, inducing saddle-node bifurcations ($\lambda_i \to +1$) to eliminate non-compliant attractors.
4. **Autoregressive Output**: Decodes tokens conditioned on the collapsed attractor $z^*_{\text{collapsed}}$.

---

## 5. Related Literature & Prior Work

1. **Universal Transformers & Looped Architectures**: Dehghani et al. (2019), Giannou et al. (2023) - Recurrent depth, looped transformers, and test-time compute iterations.
2. **Implicit Deep Equilibrium Models**: Bai et al. (2019, 2020) - Fixed-point representations and implicit differentiation.
3. **Latent Deliberation & Recurrent Depth**: Coconut (Hao et al., 2024), Saunshi et al. (2025) - Reasoning with latent thoughts in looped transformers.
4. **Attractor Networks & Dynamical Systems**: "Scaling Up Test-Time Compute with Latent Reasoning" (Geiping et al., 2025; Ouro, 2025).
