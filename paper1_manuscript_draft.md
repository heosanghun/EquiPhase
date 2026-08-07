# Paper 1 Manuscript: Auditing Data Leakage, Confounders, and Provenance Artifacts in AI for Science Benchmarks

**Author**: Sanghoon Huh (허상훈)  
**Target Journal / Venue**: Bioinformatics / BioData Mining / NeurIPS Datasets and Benchmarks Track  
**Date**: August 2026  

---

## Abstract
Machine learning benchmarks in computational biology and clinical medicine often exhibit inflated performance estimates due to hidden data leakage, baseline confounders, and undocumented dataset provenance artifacts. In this work, we propose the Unified Provenance and Audit Framework (UPAF), a cryptographic auditing framework that seals dataset provenance, split rules, runtime environments, code execution, and holdout predictions. We apply UPAF to audit two primary domain benchmarks: (1) an intrinsically disordered protein liquid-liquid phase separation (LLPS) prediction benchmark (Task B) and (2) a multi-center leave-one-site-out clinical heart disease dataset (Task F), while presenting a retrospective forensic case study on a third (Task A). We further report three real-world case studies detailing provenance leakage in our benchmark, a dataset parsing artifact, and an audit-log overwrite incident. Our empirical audits reveal that baseline sequence length alone achieves an AUROC of 0.6017 [0.5569, 0.6465], while metadata text annotation length yields an independent literature bias of 0.5762 [0.5306, 0.6218]. In multi-center clinical auditing, our 3-axis protocol indicates model discriminative power beyond demographic chance (permutation $p = 0.0010$, demographic gain $\Delta = +0.1287$) while identifying severe regional degradation at sites with missing measurements (Switzerland AUROC 0.5967).

---

## 1. Introduction
The rapid growth of AI for Science (AI4Sci) has accelerated model development across proteomics, structural biology, and digital health. However, recent retrospective evaluations have called into question whether reported performance gains reflect genuine biological learning or subtle shortcuts exploited by high-capacity neural networks. Data leakage—whether arising from shared sequence homology across splits, unreported metadata artifacts, or demographic confounders—remains a pervasive barrier to clinical and biological deployment.

In this work, we make three main contributions:
1. **The UPAF Cryptographic Seal Protocol**: A 5-layer hash sealing and append-only audit framework for benchmarks.
2. **The 3-Axis Audit Evaluation Protocol**: Disentangling model learning from demographic and confounder baselines.
3. **Empirical Benchmarking Case Studies**: Uncovering sequence length bias, FASTA parsing artifacts, multi-site clinical degradation, and audit log immutability lessons.

---

## 2. The UPAF Cryptographic Auditing Framework
UPAF establishes a 5-layer cryptographic seal:
1. **Data Layer**: SHA-256 canonical array and raw file hashing (`X`, `y`, `confounds`, `raw_files`).
2. **Split Layer**: Integer-pinned fold indices and split policy pinning.
3. **Code & Environment Layer**: Exact entry script hashing, module state, and runtime dependencies.
4. **Execution Layer**: Model configuration, hyperparameter seeds, and metric definitions.
5. **Output Layer**: Raw holdout prediction persistence (`sample_id`, `y_true`, `y_score`, `y_pred`).

Post-GENESIS audit records link sequentially via cryptographic hash chaining (`prev_manifest_self_sha256`) anchored by an external repository tip hash (`ledger_tip.sha256`). Records created before the GENESIS migration checkpoint predate the current serialization convention and are classified as `unverifiable_legacy` (Appendix A).

*System Boundary & Scope*: The seal detects post-hoc modification and rerun inconsistency. It does not detect confounds in the data itself, nor manipulation at execution time; the former is addressed by the 3-axis protocol.

---

## 3. The 3-Axis Audit Evaluation Protocol
To distinguish genuine model learning from artifact exploitation, holdout predictions are subjected to a 3-axis evaluation:
- **Axis B1 (Permutation Significance)**: Shuffling labels and re-training models ($1,000+$ fits per fold) to construct empirical null distributions.
- **Axis B2 (Demographic Baseline Gain)**: Comparing holdout predictions against leak-free demographic baselines (e.g., Age baseline). *Note: Axis B2 evaluates Demographic Gain; Matched-Confound Decoy Generators are designated for Future Work.*
- **Axis B3 (Score-Confound Correlation)**: Quantifying residual correlation between prediction scores and demographic variables.

---

## 4. Empirical Audits across Domain Benchmarks

### 4.1 Sequence Length Confound and Independent Annotation Bias in LLPS Benchmarks (Task B)
Auditing the canonical $n=697$ human intrinsically disordered protein dataset (`val.tsv`, sourced from LLPSDB / PhaSepDB; You et al., 2020) revealed that 41 rows contained missing sequence strings (`"UNKNOWN"`). Isolating the valid $n=656$ sequence cohort yielded the following sealed metrics:

| Metric Identifier | Target Feature Evaluated | Sample Size ($n$) | AUROC | 95% Confidence Interval | Null Value ($0.50$) Included |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`CONF_seqlen`** | Pure AA Sequence Length | $656$ | **`0.6017`** | `[0.5569, 0.6465]` | **No** (Statistically Significant) |
| **`CONF_seqlen (≥30 AA)`** | Sensitivity Subset (Length $\ge 30$) | $648$ | **`0.5996`** | `[0.5544, 0.6448]` | **No** (Statistically Significant) |
| **`CONF_header`** | FASTA Header Text Length | $656$ | **`0.5762`** | `[0.5306, 0.6218]` | **No** (Statistically Significant) |
| **`CONF_missing`** | Missing Sequence String Indicator | $697$ | **`0.4892`** | `[0.4436, 0.5348]` | **Yes** (Uninformative Noise) |

### 4.2 Multi-Site Clinical Generalization and Leak-Free Audit Protocol (Task F)
Auditing the combined UCI Heart Disease dataset ($n=920$, 4 clinical sites; Detrano et al., 1989) under leak-free fold median imputation and Leave-One-Site-Out (LOSO) validation yielded:

| Clinical Site Identifier | Geographic Origin | Positive Cohort ($y=1$) | Negative Cohort ($y=0$) | Total Cohort ($n_i$) | Holdout AUROC | Holdout Age Baseline (Raw Log) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Cleveland** | USA (Ohio) | $139$ | $164$ | $303$ | **`0.7902`** | `0.6370` |
| **Hungarian** | Hungary (Budapest) | $106$ | $188$ | $294$ | **`0.7956`** | `0.5868` |
| **Switzerland** | Switzerland (Zurich) | $115$ | $8$ | $123$ | **`0.5967`** | `0.5413` |
| **VA Long Beach** | USA (California) | $149$ | $51$ | $200$ | **`0.6939`** | `0.5967` |
| **Combined (Mean $\pm$ SD)** | Multi-Center | $509$ | $411$ | $920$ | **`0.7191 ± 0.0814`** | **`0.5904 ± 0.0340`** |

#### 3-Axis Protocol Evaluation Results:
- **Axis B1**: 1,000 re-trained permutation fits per fold ($4,000$ total fits) produced an empirical null mean of $0.5023 \pm 0.0615$ (observed site-specific null SDs: Cleveland $0.1449$, Hungarian $0.1490$, Switzerland $0.0904$, VA Long Beach $0.0967$). The theoretical 4-fold combined null SD under fold independence, $\frac{\sqrt{0.1449^2 + 0.1490^2 + 0.0904^2 + 0.0967^2}}{4} = \mathbf{0.0616}$, closely matches the observed 4-fold mean null SD ($0.0615$) to 3 decimal places, confirming fold independence. Separately, the holdout permutation $p$-value confirms statistically significant model learning ($p = 0.0010$).
- **Axis B2**: Gain over demographic Age baseline ($0.5904 \pm 0.0340$) was $\mathbf{\Delta AUROC = +0.1287}$.
- **Axis B3**: Prediction-age score correlation was $r = 0.3889$ ($p = 1.39 \times 10^{-34}$).

---

## 5. Case Studies of Benchmark Artifacts and Provenance Leakage

### 5.1 Case Study 1: Provenance Leakage in Fold-Switching Protein Pair Benchmarks (Task A)
In auditing the fold-switching protein benchmark ($n=156$, 93 switchers / 63 controls; Chakravarty & Porter 2022; SHA-256 `7fdd599046...`), the following three artifacts were identified through manual code forensics rather than automated framework scanning (which subsequently motivated adding provenance tracking and source-only baseline checks to UPAF):
1. **Source-Label Confounding**: Positive pairs were sourced from Chakravarty & Porter (2022) whereas negative controls were constructed from UniProt, embedding non-biophysical dataset generation artifacts into label assignments.
2. **Raw Structural Alignment Discrimination**: A simple, un-trained raw structural RMSD baseline achieved an AUROC of **`0.7981`** without learning protein dynamics.
3. **Residue Indexing Misalignment**: Calculating RMSD across residue indices without sequence alignment caused control-pair RMSD values to reach a median of **`12.6 Å`**, far above the $0.5\text{--}2\text{ \AA}$ typical of same-fold pairs. Because alignment errors artificially inflated control RMSD values toward the fold-switching range, proper sequence alignment is expected to further increase the baseline discriminative RMSD AUROC above 0.7981.

### 5.2 Case Study 2: FASTA Text String Parsing Artifacts (Task B)
Auditing the raw `val.tsv` file revealed that 41 missing sequence rows were initially converted into 5-residue peptide strings (`NKNWN`) due to naive regex cleaning of `"UNKNOWN"` text cells. Excluding these rows isolated true sequence length bias (`CONF_seqlen = 0.6017`) and header text length bias (`CONF_header = 0.5762`).

### 5.3 Case Study 3: Audit Log Overwrite Incident and Hash Chaining Requirements (UPAF Incident)
During audit log maintenance, a script (`rewrite_invalidations.py`) executed in write mode (`"w"`) rather than append mode (`"a"`), resulting in the loss of 5 invalidation log entries (prior log state SHA-256 `b3f12e56...`). Cryptographic hash chaining (`prev_manifest_self_sha256`) was subsequently designed and introduced specifically in response to this incident. Pre-GENESIS ledger records (lines 1-14) remain designated as `unverifiable_legacy`. Furthermore, because internal hash chaining alone cannot prevent whole-file rewrites, external tip anchoring (`ledger_tip.sha256`) in git history is essential to guarantee audit log immutability.

---

## 6. Reproduction and Audit Guidelines
1. **Cryptographic Seal Enforcement**: All benchmark evaluations should seal data, split, code, execution, and holdout prediction layers.
2. **Explicit Baseline Confound Reporting**: Model AUROCs must be accompanied by non-biophysical feature baselines (e.g., sequence length, age).
3. **Re-training Permutation Protocol**: Permutation tests must re-fit model weights ($1,000+$ fits per fold) to test learning capacity rather than fixed score variance.
4. **Append-Only Ledger Enforcement**: Audit logs must never be opened in write mode (`"w"`) by any script. All modifications or corrections must be logged as new appended records.
5. **Non-Mutating Inspection Protocol**: Audit log verification must be performed by a read-only inspector (`verify_only`) that never appends side-effect records during inspection.
6. **External Tip Anchoring**: Operational logs must be anchored out-of-band via external SHA-256 tip files integrated into git history.

---

## 7. Conclusion
Establishing trustworthy AI for Science and Healthcare requires transparent, tamper-proof auditing mechanisms that go beyond reporting raw accuracy metrics. The UPAF framework and 3-axis protocol provide a practical blueprint for identifying baseline confounders, structural alignment artifacts, and dataset leakage. By sealing benchmark provenance and enforcing immutable audit logs, research teams can ensure that reported AI advances reflect true scientific discovery.

---

## Appendix A. Known System Limitations
1. **Pre-GENESIS Audit Logs**: Ledger entries 1-14 created prior to the GENESIS migration checkpoint are classified as `unverifiable_legacy` due to initial serialization variations.
2. **Operational Basin Sampling**: In high-dimensional representation spaces ($d \approx 3000$), candidate attractor sampling is relative to a specified finite set of initializations $\mathcal{Z}_0$.

---

## References
- Bai, S., Kolter, J. Z., & Koltun, V. (2019). Deep equilibrium models. *Advances in Neural Information Processing Systems*, 32.
- Chakravarty, D., & Porter, L. L. (2022). AlphaFold2 fails to predict protein fold switching. *Protein Science*, 31(6), e4353. https://doi.org/10.1002/pro.4353
- Dehghani, M. et al. (2019). Universal transformers. *ICLR*.
- Detrano, R. et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*, 64(5), 304-310.
- Geiping, J. et al. (2025). Scaling up test-time compute with latent reasoning. *arXiv:2502.05171*.
- Giannou, A. et al. (2023). Looped transformers as programmable computers. *ICML*.
- Hao, L. et al. (2024). Training large language models to reason in a continuous latent space. *arXiv:2412.06769*.
- Krotov, D., & Hopfield, J. J. (2016). Dense associative memory for pattern recognition. *NIPS*.
- Ramsauer, H. et al. (2020). Hopfield networks is all you need. *ICLR*.
- Saunshi, N. et al. (2025). Reasoning with latent thoughts: On the power of looped transformers. *arXiv:2502.17416*.
- You, K. et al. (2020). PhaSepDB: a database of phase separation proteins. *Nucleic Acids Research*, 48(D1), D388-D395.
- Zhu, Y. et al. (2025). Ouro: Recurrent depth language models. *arXiv:2502.04328*.
