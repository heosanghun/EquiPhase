# S-DEQ + LLM Dual-Axis Agent Architecture: Operational Multi-Stability and Measurement Metrics

**Authors**: Antigravity AI Research Team  
**Date**: August 2026  
**Document Version**: 1.0 (Formal Specification)

---

## 1. Executive Summary and Philosophical Foundation

In high-stakes agentic systems, AI architectures must reconcile two distinct forms of computation:
1. **Explicit Sequential Reasoning (Token Axis)**: Step-by-step, human-interpretable derivation via autoregressive language models (e.g., Chain-of-Thought, tool call sequences).
2. **Implicit Deep Equilibrium Compute (Latent Axis)**: Iterative fixed-point solving over continuous representation spaces, enabling input-adaptive test-time compute.

Standard Deep Equilibrium Models (DEQ) solve for a single stable equilibrium $z^* = f_\theta(z^*; x)$, discarding intermediate solver trajectories via the Implicit Function Theorem. While standard DEQ lacks readable reasoning trajectories, **Stochastic/Multi-stable Deep Equilibrium Models (S-DEQ)** introduce multi-basin energy landscapes $\mathcal{E}(z; x)$. 

In this dual-axis architecture, **Multi-stability does not represent sequential reasoning; rather, it formalizes the coexistence of candidate hypothesis basins.** At any decision epoch $t$, the number of stable energy basins $N(x_t)$ represents the set of structurally valid, non-conflicting action pathways. Operational constraints and external controls act as energy perturbations that collapse the multi-stable state into a unique execution basin.

---

## 2. Dual-Axis Architecture Formalism

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
|   Energy Function:  E(z; x_t) = 0.5 * || z - f_theta(z; x_t) ||^2                                |
|                                                                                                  |
|   Multi-Stable Equilibrium States (N-Basins):                                                     |
|       z^*_1, z^*_2, ..., z^*_N  s.t.  grad_z E(z^*_k; x_t) = 0 ,  Hessian_z E(z^*_k; x_t) > 0        |
|                                                                                                  |
|   Operational Constraint Perturbation:  u_t (Safety Policy / Guardrails)                         |
|       E_perturbed(z; x_t, u_t) = E(z; x_t) + <u_t, g(z)>                                         |
|                                                                                                  |
|   State Collapse -> Unique Equilibrium z*_(collapsed)                                            |
+--------------------------------------------------------------------------------------------------+
                                               |
                                    Collapsed Latent State z*
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

#### Definition 2 (Latent Axis Fixed-Point & Energy Function)
At decision step $t$, the context vector $x_t = \text{Embed}(y_{<t})$ conditions the continuous latent mapping $f_\theta(\cdot ; x_t): \mathbb{R}^d \to \mathbb{R}^d$. We define the potential energy landscape $\mathcal{E}(z; x_t)$ as:
$$\mathcal{E}(z; x_t) \triangleq \frac{1}{2} \| z - f_\theta(z; x_t) \|_2^2$$

#### Definition 3 (Multi-Stable Equilibrium Set)
The set of stable equilibrium basins $\mathcal{S}(x_t) = \{ z_1^*, z_2^*, \dots, z_N^* \}$ satisfies:
$$\nabla_z \mathcal{E}(z_k^*; x_t) = 0 \quad \text{and} \quad \lambda_{\min}\left( \nabla_z^2 \mathcal{E}(z_k^*; x_t) \right) > 0 \quad \forall \, z_k^* \in \mathcal{S}(x_t)$$
where $N = |\mathcal{S}(x_t)|$ defines the **Operational Multi-Stability Index**.

---

## 3. Operational Definitions of Multi-Stability

In an autonomous agent context, multi-stability is operationally defined as follows:

| Property | Standard Single-Basin DEQ | S-DEQ Multi-Stable Agent | Agentic Interpretation |
| :--- | :--- | :--- | :--- |
| **Equilibrium Count ($N$)** | $N = 1$ (Mono-stable) | $N \ge 1$ (Multi-stable) | Number of valid candidate action paths |
| **Trajectory Meaning** | Solver implementation artifact | Phase-space relaxation | Test-time compute (Input-adaptive depth) |
| **Basin Coexistence** | Single forced convergence | $N$ competing attraction basins | Coexistence of alternative hypotheses |
| **State Collapse** | Deterministic fixed point | Constraint-driven bifurcation | Policy choice & safety guardrail selection |

### 3.1 Operational Meaning of the Energy Basins
- **Basin Coordinates ($z_k^*$)**: Represents candidate action semantics (e.g., $z_1^* \sim$ "Query Database", $z_2^* \sim$ "Call API", $z_3^* \sim$ "Request Human Clarification").
- **Basin Energy Depth ($\mathcal{E}_k$)**: Represents internal self-consistency / confidence of hypothesis $k$.
- **Inter-Basin Energy Barrier ($\Delta V_{jk}$)**: Quantifies how difficult it is for the agent to switch between candidate strategies without external input.

---

## 4. Quantitative Measurement Metrics

To quantify and monitor the latent axis in production systems, we establish four primary measurement metrics:

### Metric 1: Operational Multi-Stability Index ($N_{\text{basins}}$)
$$N_{\text{basins}}(x_t) = \text{Count}\left( \{ z \in \mathbb{R}^d \mid \| z - f_\theta(z; x_t) \|_2 < \epsilon_{\text{tol}}, \, \text{det}(I - J_f(z)) > 0 \} \right)$$
- **Interpretation**: Measures the number of distinct valid action paths available to the agent at decision step $t$. High $N_{\text{basins}}$ indicates high ambiguity / multiple viable strategies.

### Metric 2: Basin Residual Variance ($\sigma_{\text{residual}}^2$)
$$\sigma_{\text{residual}}^2 = \frac{1}{K} \sum_{k=1}^K \| z^{(k)} - f_\theta(z^{(k)}; x_t) \|_2^2$$
- **Interpretation**: Evaluates the mathematical convergence quality of the numerical solver (e.g., Anderson Acceleration or Broyden's Method) on the latent axis.

### Metric 3: Transition Barrier Height ($\Delta V_{ij}$)
$$\Delta V_{ij}(x_t) = \min_{\gamma \in \Gamma(z_i^*, z_j^*)} \max_{s \in [0,1]} \mathcal{E}(\gamma(s); x_t) - \max\left( \mathcal{E}(z_i^*; x_t), \mathcal{E}(z_j^*; x_t) \right)$$
- **Interpretation**: Measures the stability of hypothesis $i$ against spontaneous jumping to hypothesis $j$. High barriers prevent decision flickering under noisy inputs.

### Metric 4: Constraint Collapse Sensitivity ($\chi_{\text{collapse}}$)
$$\chi_{\text{collapse}}(u_t) = \frac{\partial z_{\text{collapsed}}^*}{\partial u_t} = \left( I - \nabla_z f_\theta(z^*; x_t, u_t) \right)^{-1} \cdot \frac{\partial f_\theta}{\partial u_t}$$
- **Interpretation**: Measures how efficiently external operational constraints or safety guardrail vectors $u_t$ steer the agent's multi-stable state into a safe, single-basin outcome.

---

## 5. Integration Model: Gemma 4 + S-DEQ Architecture

In an integrated deployment (e.g., Gemma 4 + S-DEQ):

1. **Gemma 4 (Token Axis)**: Processes user prompts, formats tools, and emits readable Chain-of-Thought logs.
2. **S-DEQ Module (Latent Axis)**: Embedded inside the intermediate transformer layer representations. Per-token or per-decision-step, the S-DEQ module executes fixed-point iterations:
   $$z^{(k+1)} = f_\theta(z^{(k)}; x_t)$$
   enabling deep, input-adaptive latent deliberation.
3. **Operational Constraint Layer**: Enforces safety policy vectors $u_t$. If $N_{\text{basins}} > 1$, $u_t$ modifies the potential landscape $\mathcal{E}(z; x_t, u_t)$, collapsing unsafe basins and selecting the optimal compliant action $z_{\text{collapsed}}^*$.
4. **Final Autoregressive Step**: Gemma 4 condition on $z_{\text{collapsed}}^*$ to output the final token/tool execution call.

---

## 6. Summary Specification

- **DEQ Nature**: Pure numerical solver trajectory is discarded; solver steps equal test-time compute depth.
- **CoT Nature**: Human-readable sequential derivation preserved on token axis.
- **Multi-Stability Role**: Models the coexistence and controlled collapse of $N$ valid candidate action paths.
- **Dual-Axis Synergy**: Token axis provides explicit reasoning and readability; Latent axis provides deep adaptive compute and multi-hypothesis stability.
