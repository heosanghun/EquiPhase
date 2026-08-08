# Paper 1 — Section 5 & 6 Draft

## §5 Case Studies of Benchmark Artifacts and Provenance Leakage

### 5.1 Case Study 1: Provenance Leakage in Fold-Switching Protein Pair Benchmarks (Task A)
In auditing the fold-switching protein benchmark ($n=156$, 93 switchers / 63 controls; Chakravarty & Porter 2022; SHA-256 `7fdd599046...`), we discovered that initial benchmark setups contained non-biophysical dataset ordering and pair indexing cues that allowed un-trained decision trees to achieve near-perfect classification based on line position alone. Re-indexing and shuffling by sequence identity proved essential to eliminate provenance leakage.

### 5.2 Case Study 2: FASTA Text, Unicode, and Column Parsing Artifacts (Task B)
During our audit of the human LLPS benchmark dataset (`val.tsv` and `train.tsv`), three separate text parsing artifacts were identified:
1. **Missing Sequence String Conversion**: 41 missing sequence rows in validation were initially parsed into 5-residue peptide strings (`NKNWN`) due to naively applying regex replacement (`re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', s)`) to the raw cell string `"UNKNOWN"`. Correcting this parsing artefact brought the dataset down to $n=656$ valid sequence rows, revealing that sequence length bias (`CONF_seqlen = 0.6017`) and header text length bias (`CONF_header = 0.5762`) operated independently, whereas missingness itself (`CONF_missing = 0.4892`) was uninformative noise.
2. **Unicode Non-Breaking Space Misclassification**: Organism attribute `"OS=Homo sapiens"` contained non-breaking spaces (`\xa0`) in 16 validation and 37 training rows, misclassifying human entries as non-human when matched on standard spaces.
3. **Unparseable Sequence Length Header Column**: The file's internal `'Sequence length'` column contained non-numeric text strings across all entries (`collen = None`), rendering the raw column unusable and requiring pure sequence reconstruction (`complen`).

### 5.3 Case Study 3: Audit Log Overwrite Incident and Hash Chaining Requirements (UPAF Incident)
During system development, an invalidation update script (`rewrite_invalidations.py`) executed using write mode (`"w"`) rather than append mode (`"a"`), temporarily overwriting preceding log entries. This real-world incident demonstrated a key lesson for AI auditability: **Single-file append-only logs are not append-only as long as write-capable processes exist.** To guarantee immutability, audit manifests must incorporate cryptographic hash chaining (`prev_manifest_self_sha256`), external tip anchoring (`ledger_tip.sha256`), and version control integration.

### 5.4 Case Study 4: Numerical Provenance Mapping, Unanchored Conversation Records, and Audit Reclassification
During manuscript compilation, four subgroup AUROC values were recorded in conversation logs without anchored execution traces. A 16-variant matrix audit (`claude_paper1_sec41_recompute_v3.py`, SHA-256 `925a2433...`) proved that the conversation-record strata values (0.5898 / 0.6173) were unreproducible legacy artifacts of unknown provenance, whereas previously blocked values (0.6291 / 0.5756) were reclassified as probable early unanchored calculation runs — though their exact generation variant remains unverified. This case study demonstrates UPAF's core principle: *verbal assertions and unanchored conversation records carry no canonical status; only cryptographically sealed execution outputs constitute verifiable audit evidence.*

### 5.5 Case Study 5: Audit Discrepancies in Equilibrium Neural Network Specifications
Auditing an implicit Deep Equilibrium (DEQ) network implementation (`train_paper2_deq_supervised.py`) uncovered three major protocol and implementation discrepancies:
1. **Hardcoded Log Statement Literals (Pattern 2, 4th instance)**: The reported exact 0.00% force anti-symmetry ($G_1$) in early audit logs resulted from a static string literal `print(f"[G1 Architectural Guarantee] Force Anti-Symmetry: 0.0000e+00%")` omitting format variables. While sealed execution measurements confirmed near-zero anti-symmetry ($0.0 \sim 2.63 \times 10^{-9}$), asserting claims without dynamic variable logging violated audit transparency.
2. **Locked-Specification Implementation Deviation**: While the preregistration specification required Implicit Function Theorem (IFT) exact analytical backpropagation, the actual code unrolled 100 forward solver iterations into the autograd computation graph.
3. **Evidence Inflation from Stream Truncation (Pattern 12)**: Earlier audit logs claimed 100% SHA-256 hash identity between verification runs (`run2` vs `run3`), which was subsequently traced to prematurely truncated 108-line log files. Full-length execution logs confirmed zero-diff bitwise reproducibility on clean output lines while exposing wall-clock execution differences ($25.5\text{ s}$ retrain duration).
4. **Reporting Transmission Channel Artifacts (Pattern 13)**: While cryptographically sealed execution scripts, self-hashing, and deterministic seeding guaranteed execution integrity, forensic auditing revealed that the primary vulnerability layer was the reporting transmission channel itself. Three successive reporting reconstruction discrepancies—detected via invariant parameter count canaries, structural matrix mismatches, and non-unique hash slot artifacts—demonstrated that verbal summaries and raw output text serialization remain susceptible to LLM output reconstruction artifacts. The final defense layer of a cryptographic auditing framework must therefore rely on independent human-in-the-loop channel verification.

Furthermore, historical log conflicts between Set A ($45/6/0/49$) and Set B ($16/37/29/18$) trajectory basin cross-tabulations were conclusively resolved by a sealed 3rd-party independent audit script (`claude_paper2_sealed_audit.py`, SHA-256 `68a2991e0439...`), confirming Set B as the authentic deterministic result. The sealed audit also identified a 1/100 trajectory divergence boundary for large initializations $\|z_0\|$, establishing global convergence limits for damped velocity Verlet integration.

---

## §6 Reproduction, Integrity, and Audit Guidelines

1. **Cryptographic Seal Enforcement**: Every experimental run must record canonical serialization hashes of raw dataset files, split indices, local module code, environment runtimes, and predictions.
2. **Explicit Null Control Baseline Reporting**: Reported model metrics must be contrasted against baseline confounders (e.g., sequence length, demographic age) and holdout permutation null distributions.
3. **Re-training Permutation Protocol**: Permutation tests must re-train the model on shuffled labels ($1,000+$ fits per fold) to evaluate true model learning capacity rather than fixed score artifacts.
4. **External Tip Anchoring**: Log integrity must be anchored outside the operational process boundary (e.g., in git commit history or separate read-only storage).
