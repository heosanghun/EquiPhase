# Paper 1 — Section 5 & 6 Draft

## §5 Case Studies of Benchmark Artifacts and Provenance Leakage

### 5.1 Case Study 1: Provenance Leakage in Fold-Switching Protein Pair Benchmarks (Task A)
In auditing the fold-switching protein benchmark ($n=156$, 93 switchers / 63 controls; Chakravarty & Porter 2022; SHA-256 `7fdd599046...`), we discovered that initial benchmark setups contained non-biophysical dataset ordering and pair indexing cues that allowed un-trained decision trees to achieve near-perfect classification based on line position alone. Re-indexing and shuffling by sequence identity proved essential to eliminate provenance leakage.

### 5.2 Case Study 2: FASTA Text String Parsing Artifacts (Task B)
During our audit of the human LLPS benchmark dataset (`val.tsv`), 41 missing sequence rows were initially parsed into 5-residue peptide strings (`NKNWN`) due to naively applying regex replacement (`re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', s)`) to the raw cell string `"UNKNOWN"`. Correcting this parsing artefact brought the dataset down to $n=656$ valid sequence rows, revealing that sequence length bias (`CONF_seqlen = 0.6017`) and header text length bias (`CONF_header = 0.5762`) operated independently, whereas missingness itself (`CONF_missing = 0.4892`) was uninformative noise.

### 5.3 Case Study 3: Audit Log Overwrite Incident and Hash Chaining Requirements (UPAF Incident)
During system development, an invalidation update script (`rewrite_invalidations.py`) executed using write mode (`"w"`) rather than append mode (`"a"`), temporarily overwriting preceding log entries. This real-world incident demonstrated a key lesson for AI auditability: **Single-file append-only logs are not append-only as long as write-capable processes exist.** To guarantee immutability, audit manifests must incorporate cryptographic hash chaining (`prev_manifest_self_sha256`), external tip anchoring (`ledger_tip.sha256`), and version control integration.

### 5.5 Case Study 5: Audit Discrepancies in Equilibrium Neural Network Specifications
Auditing an implicit Deep Equilibrium (DEQ) network implementation (`train_paper2_deq_supervised.py`) uncovered two major protocol and implementation discrepancies:
1. **Hardcoded Log Statement Literals (Pattern 2, fourth documented instance)**: The reported exact 0.00% force anti-symmetry ($G_1$) in early audit logs resulted from a static string literal `print(f"[G1 Architectural Guarantee] Force Anti-Symmetry: 0.0000e+00%")` containing no format variables, obscuring true floating-point precision dynamics ($\sim 1.646 \times 10^{-10}$).
2. **Locked-Specification Implementation Deviation**: While the preregistration specification required Implicit Function Theorem (IFT) exact analytical backpropagation, the actual code unrolled 100 forward solver iterations into the autograd computation graph.

Furthermore, audit logs for trajectory basin cross-tabulations contained conflicting historical records (Set A: 45/6/0/49 vs Set B: 16/37/29/18), which were provisionally adjudicated by a sealed independent-authorship script (execution pending raw-stdout verification) confirming Set B as the true deterministic result of seed-7777 initializations.

---

## §6 Reproduction, Integrity, and Audit Guidelines

1. **Cryptographic Seal Enforcement**: Every experimental run must record canonical serialization hashes of raw dataset files, split indices, local module code, environment runtimes, and predictions.
2. **Explicit Null Control Baseline Reporting**: Reported model metrics must be contrasted against baseline confounders (e.g., sequence length, demographic age) and holdout permutation null distributions.
3. **Re-training Permutation Protocol**: Permutation tests must re-train the model on shuffled labels ($1,000+$ fits per fold) to evaluate true model learning capacity rather than fixed score artifacts.
4. **External Tip Anchoring**: Log integrity must be anchored outside the operational process boundary (e.g., in git commit history or separate read-only storage).
