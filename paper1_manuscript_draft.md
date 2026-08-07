# Paper 1 Manuscript: Auditing Data Leakage, Confounders, and Provenance Artifacts in AI for Science Benchmarks

**Author**: Sanghoon Huh (허상훈)  
**Target Journal / Venue**: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) / BioData Mining / Bioinformatics  
**Date**: August 2026  

---

## Abstract
Machine learning benchmarks in computational biology and clinical medicine often exhibit inflated performance estimates due to hidden data leakage, baseline confounders, and undocumented dataset provenance artifacts. In this work, we propose the Unified Protocol for Auditability and Fairness (UPAF), a cryptographic auditing framework that seals dataset provenance, split rules, runtime environments, code execution, and holdout predictions. We apply UPAF to audit three canonical domain benchmarks: (1) an intrinsically disordered protein liquid-liquid phase separation (LLPS) prediction benchmark (Task B), (2) a multi-center leave-one-site-out clinical heart disease dataset (Task F), and (3) a fold-switching protein structure benchmark (Task A). Our empirical audits reveal that baseline sequence length alone achieves an AUROC of 0.6017 [0.5569, 0.6465], while metadata text annotation length yields an independent literature bias of 0.5762 [0.5306, 0.6218]. In multi-center clinical auditing, our 3-axis protocol proves that model discriminative power is genuine (permutation $p = 0.0010$, demographic gain $\Delta = +0.1287$) despite substantial inter-center variance ($\pm 0.0814$). Finally, we present three real-world case studies detailing provenance leakage, dataset parsing artifacts, and audit log immutability lessons.

---

## 1. Introduction
The rapid growth of AI for Science (AI4Sci) has accelerated model development across proteomics, structural biology, and digital health. However, recent retrospective evaluations have called into question whether reported performance gains reflect genuine biological learning or subtle shortcuts exploited by high-capacity neural networks. Data leakage—whether arising from shared sequence homology across splits, unreported metadata artifacts, or demographic confounders—remains a pervasive barrier to clinical and biological deployment.

---

## 2. The UPAF Cryptographic Auditing Framework
UPAF establishes a 5-layer cryptographic seal:
1. **Data Layer**: SHA-256 canonical array and raw file hashing (`X`, `y`, `confounds`, `raw_files`).
2. **Split Layer**: Integer-pinned fold indices and split policy pinning.
3. **Code & Environment Layer**: Exact entry script hashing, module state, and runtime dependencies.
4. **Execution Layer**: Model configuration, hyperparameter seeds, and metric definitions.
5. **Output Layer**: Raw holdout prediction persistence (`sample_id`, `y_true`, `y_score`, `y_pred`).

All audit records link sequentially via cryptographic hash chaining (`prev_manifest_self_sha256`) anchored by an external repository tip hash (`ledger_tip.sha256`).

---

## 3. The 3-Axis Audit Evaluation Protocol
To distinguish genuine model learning from artifact exploitation, holdout predictions are subjected to a 3-axis evaluation:
- **Axis B1 (Permutation Significance)**: Shuffling labels and re-training models ($1,000+$ fits per fold) to construct empirical null distributions.
- **Axis B2 (Demographic Baseline Gain)**: Comparing holdout predictions against leak-free demographic baselines (e.g., Age baseline).
- **Axis B3 (Score-Confound Correlation)**: Quantifying residual correlation between prediction scores and demographic variables.

---

## 4. Empirical Audits across Domain Benchmarks

### 4.1 Sequence Length Confound and Independent Annotation Bias in LLPS Benchmarks (Task B)
Auditing the canonical $n=697$ human intrinsically disordered protein dataset (`val.tsv`) revealed that 41 rows contained missing sequence strings (`"UNKNOWN"`). Isolating the valid $n=656$ sequence cohort yielded the following sealed metrics:

| Metric Identifier | Target Feature Evaluated | Sample Size ($n$) | AUROC | 95% Confidence Interval | Null Value ($0.50$) Included |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`CONF_seqlen`** | Pure AA Sequence Length | $656$ | **`0.6017`** | `[0.5569, 0.6465]` | **No** (Statistically Significant) |
| **`CONF_seqlen (≥30 AA)`** | Sensitivity Subset (Length $\ge 30$) | $648$ | **`0.5996`** | `[0.5544, 0.6448]` | **No** (Statistically Significant) |
| **`CONF_header`** | FASTA Header Text Length | $656$ | **`0.5762`** | `[0.5306, 0.6218]` | **No** (Statistically Significant) |
| **`CONF_missing`** | Missing Sequence String Indicator | $697$ | **`0.4892`** | `[0.4436, 0.5348]` | **Yes** (Uninformative Noise) |

### 4.2 Multi-Site Clinical Generalization and Leak-Free Audit Protocol (Task F)
Auditing the combined UCI Heart Disease dataset ($n=920$, 4 clinical sites) under leak-free fold median imputation and Leave-One-Site-Out (LOSO) validation yielded:

| Clinical Site Identifier | Geographic Origin | Positive Cohort ($y=1$) | Negative Cohort ($y=0$) | Total Cohort ($n_i$) | Holdout AUROC | Holdout Age Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Cleveland** | USA (Ohio) | $139$ | $164$ | $303$ | **`0.7902`** | `0.6370` |
| **Hungarian** | Hungary (Budapest) | $106$ | $188$ | $294$ | **`0.7956`** | `0.6133` |
| **Switzerland** | Switzerland (Zurich) | $115$ | $8$ | $123$ | **`0.5967`** | `0.5489` |
| **VA Long Beach** | USA (California) | $149$ | $51$ | $200$ | **`0.6939`** | `0.5623` |
| **Combined (Mean $\pm$ SD)** | Multi-Center | $509$ | $411$ | $920$ | **`0.7191 ± 0.0814`** | **`0.5904 ± 0.0354`** |

#### 3-Axis Protocol Evaluation Results:
- **Axis B1**: 1,000 re-trained permutation fits per fold ($4,000$ total fits) produced an empirical null mean of $0.5023 \pm 0.0615$ (observed site-specific null SDs: Cleveland $0.1449$, Hungarian $0.1490$, Switzerland $0.0904$, VA Long Beach $0.0967$). The theoretical 4-fold combined null SD under fold independence, $\frac{\sqrt{0.1449^2 + 0.1490^2 + 0.0904^2 + 0.0967^2}}{4} = \mathbf{0.0616}$, matches the observed 4-fold mean null SD ($0.0615$) to 4 decimal places, confirming fold independence ($p = 0.0010$).
- **Axis B2**: Gain over demographic Age baseline ($0.5904 \pm 0.0354$) was $\mathbf{\Delta AUROC = +0.1287}$.
- **Axis B3**: Prediction-age score correlation was $r = 0.3889$ ($p = 1.39 \times 10^{-34}$).

---

## 5. Case Studies of Benchmark Artifacts and Provenance Leakage

### 5.1 Case Study 1: Provenance Leakage in Fold-Switching Protein Pair Benchmarks (Task A)
In auditing the fold-switching protein benchmark ($n=156$, 93 switchers / 63 controls; Chakravarty & Porter 2022; SHA-256 `7fdd599046...`), non-biophysical ordering cues were identified, demonstrating that un-trained classifiers could exploit dataset index position unless randomized by sequence identity.

### 5.2 Case Study 2: FASTA Text String Parsing Artifacts (Task B)
Auditing the raw `val.tsv` file revealed that 41 missing sequence rows were initially converted into 5-residue peptide strings (`NKNWN`) due to naive regex cleaning of `"UNKNOWN"` text cells. Excluding these rows isolated true sequence length bias (`CONF_seqlen = 0.6017`) and header text length bias (`CONF_header = 0.5762`).

### 5.3 Case Study 3: Audit Log Overwrite Incident and Hash Chaining Requirements (UPAF Incident)
During audit log maintenance, a script executed in write mode (`"w"`) rather than append mode (`"a"`), demonstrating that single-file logs require cryptographic hash chaining (`prev_manifest_self_sha256`) and external repository tip anchoring (`ledger_tip.sha256`) to guarantee audit immutability.

---

## 6. Reproduction and Audit Guidelines
1. **Cryptographic Seal Enforcement**: All benchmark evaluations should seal data, split, code, execution, and holdout prediction layers.
2. **Explicit Baseline Confound Reporting**: Model AUROCs must be accompanied by non-biophysical feature baselines (e.g., sequence length, age).
3. **Re-training Permutation Protocol**: Permutation tests must re-fit model weights ($1,000+$ fits per fold) to test learning capacity rather than fixed score variance.
4. **External Tip Anchoring**: Operational logs must be anchored out-of-band via external SHA-256 tip files integrated into git history.

---

## Appendix A. Known System Limitations
1. **Pre-GENESIS Audit Logs**: Ledger entries 1-14 created prior to the GENESIS migration checkpoint are classified as `unverifiable_legacy` due to initial serialization variations.
2. **Operational Basin Sampling**: In high-dimensional representation spaces ($d \approx 3000$), candidate attractor sampling is relative to a specified finite set of initializations $\mathcal{Z}_0$.

---

## References
- Bai, S., Kolter, J. Z., & Koltun, V. (2019). Deep equilibrium models. *Advances in Neural Information Processing Systems*, 32.
- Chakravarty, A. & Porter, L. L. (2022). Benchmarking fold-switching protein prediction algorithms. *Structure*, 30(6), 845-857.
- Dehghani, M. et al. (2019). Universal transformers. *ICLR*.
- Detrano, R. et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*, 64(5), 304-310.
- Geiping, J. et al. (2025). Solve the loop: Attractor models for language and reasoning. *arXiv preprint*.
- Pan, L. et al. (2024). Training large language models to reason in a continuous latent space. *arXiv preprint*.
- Saunshi, N. et al. (2025). Reasoning capabilities of looped transformers. *arXiv preprint*.
