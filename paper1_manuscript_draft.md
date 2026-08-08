# Paper 1 Manuscript: Auditing Data Leakage, Confounders, and Provenance Artifacts in AI for Science Benchmarks

**Author**: Sanghoon Huh (허상훈)  
**Target Journal / Venue**: Bioinformatics / BioData Mining / NeurIPS Datasets and Benchmarks Track  
**Date**: August 2026  

---

## Abstract
Machine learning benchmarks in computational biology and clinical medicine often exhibit inflated performance estimates due to hidden data leakage, baseline confounders, and undocumented dataset provenance artifacts. In this work, we propose the Unified Provenance and Audit Framework (UPAF), a cryptographic auditing framework that seals dataset provenance, split rules, runtime environments, code execution, and holdout predictions. We apply UPAF to audit two primary domain benchmarks: (1) a multi-species intrinsically disordered protein liquid-liquid phase separation (LLPS) prediction benchmark (Task B) and (2) a multi-center leave-one-site-out clinical heart disease dataset (Task F), while presenting a retrospective forensic case study on a third (Task A). We further report four real-world case studies detailing provenance leakage in our benchmark, dataset parsing artifacts (including non-breaking space Unicode misclassification), an audit-log overwrite incident, and numerical manuscript provenance failures. Our empirical audits reveal that baseline sequence length alone achieves an AUROC of 0.6017 [0.5569, 0.6465] on the validation split ($n=656$) and 0.6010 [0.5781, 0.6239] on the training split ($n=2539$), while metadata text annotation length yields an independent literature bias of 0.5762 [0.5306, 0.6218]. Evaluating sequence length bias across taxonomic strata within the training split ($n=2539$) shows consistent bias across both non-human organisms (AUROC $0.6173$ [0.5833, 0.6508]) and human proteins (AUROC $0.5898$ [0.5574, 0.6206]), reaching a peak of $0.6406$ [0.5654, 0.7107] in viral SARS-CoV-2 proteins ($n=253$). The difference ($\Delta = 0.0275$) is within the comparison resolution limit ($\text{MDD} = 0.0465$), confirming no statistically significant taxonomic interaction ($p > 0.05$). In multi-center clinical auditing, our 3-axis protocol indicates model discriminative power beyond demographic chance (permutation $p = 0.0010$, demographic gain $\Delta = +0.1287$) while identifying severe regional degradation at sites with missing measurements (Switzerland AUROC 0.5967).

---

## 1. Introduction
The rapid growth of AI for Science (AI4Sci) has accelerated model development across proteomics, structural biology, and digital health. However, recent retrospective evaluations have called into question whether reported performance gains reflect genuine biological learning or subtle shortcuts exploited by high-capacity neural networks. Data leakage—whether arising from shared sequence homology across splits, unreported metadata artifacts, or demographic confounders—remains a pervasive barrier to clinical and biological deployment.

In this work, we make three main contributions:
1. **The UPAF Cryptographic Seal Protocol**: A 5-layer hash sealing and append-only audit framework for benchmarks.
2. **The 3-Axis Audit Evaluation Protocol**: Disentangling model learning from demographic and confounder baselines.
3. **Empirical Benchmarking Case Studies**: Uncovering sequence length bias, taxonomic interaction, FASTA and Unicode parsing artifacts, multi-site clinical degradation, audit log immutability lessons, and numerical manuscript provenance checks.

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

### 4.1 Sequence Length Confound, Taxonomic Strata, and Annotation Bias in LLPS Benchmarks (Task B)
Auditing the multi-species intrinsically disordered protein dataset (`val.tsv` raw $n=697$, SHA-256 `b7ef171d...`; `train.tsv` raw $n=2554$, 3,546 KB, SHA-256 `ecda4a8b...`, constructed by joining PhaSepDB `PSID` records; You et al., 2020 and LLPSDB `LLPS` entry headers; Li et al., 2020) revealed a significant distribution shift in organism origin. Standardizing UniProt species mnemonics (`HUMAN`, `MYCTU`, `CAEEL`, `SARS2`, `YEAST`, etc.) and resolving Unicode non-breaking space variants (`Homo\xa0sapiens`) established exact species distributions:
- **Validation Split ($n=656$ valid headers)**: 73.63% *Homo sapiens* (483 proteins: 467 standard space + 16 non-breaking space), 26.37% non-human across 17 model organisms (173 proteins: *S. cerevisiae* 5.9%, *C. reinhardtii* 3.0%, *M. tuberculosis* 2.6%, *X. laevis* 2.3%, *D. melanogaster* 2.1%, *R. norvegicus* 2.1%, *M. musculus* 1.2%, etc.).
- **Training Split ($n=2539$ valid headers)**: 51.28% *Homo sapiens* (1302 proteins: 1265 standard space + 37 non-breaking space), 48.72% non-human across 32 model organisms (1237 proteins: SARS-CoV-2 9.97% [$n=253$], *E. coli* 4.81%, *C. elegans* 4.14%, Yeast 3.66%, *M. musculus* 2.72%, *D. melanogaster* 2.25%, etc.).

Evaluating species assignment itself as a baseline feature (`CONF_species`, binary `Is_Human` predictor vs phase separation label) using 2,000-iteration stratified bootstrap CIs yielded AUROCs of **`0.4872` [0.4398, 0.5359]** (SE = $0.0245$) on validation ($n=656$) and **`0.4953` [0.4705, 0.5186]** (SE = $0.0123$) on training ($n=2539$), establishing that species identity is uninformative noise ($0.50 \in \text{CI}$).

To test whether sequence length bias varies across taxonomic boundaries, we performed a stratified interaction audit across species strata within both splits (anchored by `task-1262.log`, SHA-256 `f4a0436729f45f1409ec6bcb2fc48f25dbfccf6a2047ff3014cf7b1555767726`):
- **Training Split (`train.tsv` $n=2539$)**:
  - `Homo sapiens` Subset ($n=1302$): AUROC = **`0.5898`** [0.5574, 0.6206] (Bootstrap SE = $0.0162$)
  - Non-Human Subset ($n=1237$): AUROC = **`0.6173`** [0.5833, 0.6508] (Bootstrap SE = $0.0173$)
  - Viral `SARS-CoV-2` Subset ($n=253$): AUROC = **`0.6406`** [0.5654, 0.7107] (Bootstrap SE = $0.0366$)
- **Validation Split (`val.tsv` $n=656$)**:
  - `Homo sapiens` Subset ($n=483$): AUROC = **`0.5976`** [0.5442, 0.6481] (Bootstrap SE = $0.0264$)
  - Non-Human Subset ($n=173$): AUROC = **`0.6225`** [0.5317, 0.7093] (Bootstrap SE = $0.0454$)

In both splits, non-human organisms display slightly higher length bias point estimates than human proteins (train: $0.6173$ vs $0.5898$; val: $0.6225$ vs $0.5976$). However, the observed difference ($\Delta = 0.0275$, $\text{SE}_{\text{diff}} = 0.0237$) falls well within the minimum detectable difference ($\text{MDD} = 0.0465$), confirming that there is no statistically significant interaction between taxonomy and length bias ($p > 0.05$). Viral proteins exhibit the highest point AUROC ($0.6406$), though with wider uncertainty ($n=253$). The findings demonstrate that sequence length bias is a pervasive shortcut present across all examined taxonomic strata ($0.5898$ to $0.6406$).

| Dataset Split / Subset Evaluated | Target Feature Evaluated | Sample Size ($n$) | Positives ($y=1$) | Negatives ($y=0$) | AUROC | 95% Confidence Interval & SE Method | Confound Significance |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation Split (`val.tsv`)** | Pure AA Sequence Length (`CONF_seqlen`) | $656$ | $441$ ($67.2\%$) | $215$ ($32.8\%$) | **`0.6017`** | `[0.5569, 0.6465]` (Hanley-McNeil SE = $0.0229$) | **Statistically Significant** |
| **Validation Split (≥30 AA)** | Sensitivity Subset (Length $\ge 30$) | $648$ | $437$ ($67.4\%$) | $211$ ($32.6\%$) | **`0.5996`** | `[0.5544, 0.6448]` (Hanley-McNeil SE = $0.0230$) | **Statistically Significant** |
| **Validation Split (`val.tsv`)** | FASTA Header Text Length (`CONF_header`) | $656$ | $441$ ($67.2\%$) | $215$ ($32.8\%$) | **`0.5762`** | `[0.5306, 0.6218]` (Hanley-McNeil SE = $0.0232$) | **Statistically Significant** |
| **Validation Split (`val.tsv`)** | Missing Sequence Indicator (`CONF_missing`) | $697$ | $465$ ($66.7\%$) | $232$ ($33.3\%$) | **`0.4892`** | `[0.4436, 0.5348]` (Hanley-McNeil SE = $0.0232$) | Uninformative Noise ($0.50 \in \text{CI}$) |
| **Validation Split (`val.tsv`)** | Species Identity (`CONF_species`) | $656$ | $441$ ($67.2\%$) | $215$ ($32.8\%$) | **`0.4872`** | `[0.4398, 0.5359]` (Bootstrap SE = $0.0245$) | Uninformative Noise ($0.50 \in \text{CI}$) |
| **Validation Strata: Human** | `Homo sapiens` Subset (`CONF_seqlen`) | $483$ | $321$ ($66.5\%$) | $162$ ($33.5\%$) | **`0.5976`** | `[0.5435, 0.6481]` (Bootstrap SE = $0.0264$) | **Statistically Significant** |
| **Validation Strata: Non-Human** | Non-Human Subset (`CONF_seqlen`) | $173$ | $120$ ($69.4\%$) | $53$ ($30.6\%$) | **`0.6225`** | `[0.5317, 0.7093]` (Bootstrap SE = $0.0454$) | **Statistically Significant** |
| **Training Split (`train.tsv`)** | Pure AA Sequence Length (`CONF_seqlen`) | $2539$ | $1734$ ($68.3\%$) | $805$ ($31.7\%$) | **`0.6010`** | `[0.5781, 0.6239]` (Hanley-McNeil SE = $0.0117$) | **Statistically Significant** |
| **Training Split (`train.tsv`)** | Species Identity (`CONF_species`) | $2539$ | $1734$ ($68.3\%$) | $805$ ($31.7\%$) | **`0.4953`** | `[0.4705, 0.5186]` (Bootstrap SE = $0.0107$) | Uninformative Noise ($0.50 \in \text{CI}$) |
| **Training Strata: Human** | `Homo sapiens` Subset (`CONF_seqlen`) | $1302$ | $884$ ($67.9\%$) | $418$ ($32.1\%$) | **`0.5898`** | `[0.5574, 0.6206]` (Bootstrap SE = $0.0162$) | **Statistically Significant** |
| **Training Strata: Non-Human** | Non-Human Subset (`CONF_seqlen`) | $1237$ | $850$ ($68.7\%$) | $387$ ($31.3\%$) | **`0.6173`** | `[0.5833, 0.6508]` (Bootstrap SE = $0.0173$) | **Statistically Significant** |
| **Training Strata: Viral** | SARS-CoV-2 Subset (`CONF_seqlen`) | $253$ | $180$ ($71.1\%$) | $73$ ($28.9\%$) | **`0.6406`** | `[0.5654, 0.7107]` (Bootstrap SE = $0.0366$) | **Statistically Significant** |

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

### 5.2 Case Study 2: FASTA Text and Unicode Parsing Artifacts (Task B)
Auditing the raw `val.tsv` and `train.tsv` files revealed two separate text parsing artifacts:
1. **Missing Sequence String Conversion**: 41 missing sequence rows in validation were initially converted into 5-residue peptide strings (`NKNWN`) due to naive regex cleaning of `"UNKNOWN"` text cells. Excluding these rows isolated true sequence length bias (`CONF_seqlen = 0.6017`) and header text length bias (`CONF_header = 0.5762`).
2. **Unicode Non-Breaking Space Misclassification**: The organism attribute `"OS=Homo sapiens"` contained non-breaking space characters (`\xa0`) in 16 validation rows and 37 training rows. Naive string matching on standard spaces caused these rows to be misclassified as non-human, undercounting the human proportion by $2.4\%p$ (reporting 71.19% instead of true 73.63% in validation).

### 5.3 Case Study 3: Audit Log Overwrite Incident and Hash Chaining Requirements (UPAF Incident)
During audit log maintenance, a script (`rewrite_invalidations.py`) executed in write mode (`"w"`) rather than append mode (`"a"`), resulting in the loss of 5 invalidation log entries (prior log state SHA-256 `b3f12e56...`). Cryptographic hash chaining (`prev_manifest_self_sha256`) was subsequently designed and introduced specifically in response to this incident. Pre-GENESIS ledger records (lines 1-14) remain designated as `unverifiable_legacy`. Furthermore, because internal hash chaining alone cannot prevent whole-file rewrites, external tip anchoring (`ledger_tip.sha256`) in git history is essential to guarantee audit log immutability.

### 5.4 Case Study 4: Numerical Provenance Mapping, Recurrent Drift, and Verification Engine Failures
During manuscript compilation, four subgroup AUROC values were repeatedly reported with unverified numbers across multiple revisions. Attempts to automate verification created a flawed inspector that performed circular verification by generating temporary logs in the same execution cycle as inspecting them, while failing to check point-in-CI mathematical sanity constraints ($lo \le pval \le hi$). This meta-audit failure led to establishing strict immutability guidelines: verification inspectors must operate in read-only mode against immutable historical log files anchored by SHA-256 hashes, enforcement scripts must strictly validate point-in-CI mathematical constraints, and manuscript numbers must be mapped 1-to-1 against raw execution logs.

### 5.5 Case Study 5: Audit Discrepancies in Equilibrium Neural Network Specifications
Auditing an implicit Deep Equilibrium (DEQ) network implementation (`train_paper2_deq_supervised.py`) uncovered three major protocol and implementation discrepancies:
1. **Hardcoded Log Statement Literals (Pattern 2, 4th instance)**: The reported exact 0.00% force anti-symmetry ($G_1$) in early audit logs resulted from a static string literal `print(f"[G1 Architectural Guarantee] Force Anti-Symmetry: 0.0000e+00%")` omitting format variables. While underlying physics guaranteed near-zero anti-symmetry ($\sim 1.646 \times 10^{-10}$), asserting claims without dynamic variable logging violated audit transparency.
2. **Locked-Specification Implementation Deviation**: While the preregistration specification required Implicit Function Theorem (IFT) exact analytical backpropagation, the actual code unrolled 100 forward solver iterations into the autograd computation graph.
3. **Evidence Inflation from Stream Truncation (Pattern 12)**: Earlier audit logs claimed 100% SHA-256 hash identity between verification runs (`run2` vs `run3`), which was subsequently traced to prematurely truncated 108-line log files. Full-length execution logs confirmed zero-diff bitwise reproducibility on clean output lines while exposing wall-clock execution differences ($25.5\text{ s}$ retrain duration).

Furthermore, historical log conflicts between Set A ($45/6/0/49$) and Set B ($16/37/29/18$) trajectory basin cross-tabulations were conclusively resolved by a sealed 3rd-party independent audit script (`claude_paper2_sealed_audit.py`, SHA-256 `68a2991e0439...`), confirming Set B as the authentic deterministic result. The sealed audit also identified a 1/100 trajectory divergence boundary for large initializations $\|z_0\|$, establishing global convergence limits for damped velocity Verlet integration.

---

## 6. Reproduction and Audit Guidelines
1. **Cryptographic Seal Enforcement**: All benchmark evaluations should seal data, split, code, execution, and holdout prediction layers.
2. **Explicit Baseline Confound Reporting**: Model AUROCs must be accompanied by non-biophysical feature baselines (e.g., sequence length, age).
3. **Re-training Permutation Protocol**: Permutation tests must re-fit model weights ($1,000+$ fits per fold) to test learning capacity rather than fixed score variance.
4. **Append-Only Ledger Enforcement**: Audit logs must never be opened in write mode (`"w"`) by any script. All modifications or corrections must be logged as new appended records.
5. **Non-Mutating Inspection Protocol**: Audit log verification must be performed by a read-only inspector (`verify_only`) that never appends side-effect records during inspection.
6. **External Tip Anchoring**: Operational logs must be anchored out-of-band via external SHA-256 tip files integrated into git history.
7. **Strict Read-Only Log-Provenance Enforcement**: Verification engines must never create or overwrite execution logs, must verify against immutable historical log files anchored by SHA-256 hashes, and must enforce mathematical point-in-CI sanity constraints ($lo \le pval \le hi$).

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
- Li, Q. et al. (2020). LLPSDB: a database of proteins undergoing liquid-liquid phase separation in vitro. *Nucleic Acids Research*, 48(D1), D320-D327. https://doi.org/10.1093/nar/gkz780
- Ramsauer, H. et al. (2020). Hopfield networks is all you need. *ICLR*.
- Saunshi, N. et al. (2025). Reasoning with latent thoughts: On the power of looped transformers. *arXiv:2502.17416*.
- You, K., Huang, Q., Yu, C., Shen, B., Sevilla, C., Shi, M., Hermjakob, H., Chen, Y., & Li, T. (2020). PhaSepDB: a database of liquid–liquid phase separation related proteins. *Nucleic Acids Research*, 48(D1), D354-D359. https://doi.org/10.1093/nar/gkz847
- Zhu, Y. et al. (2025). Scaling latent reasoning via looped language models. *arXiv preprint*.
