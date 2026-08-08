# UPAF Cryptographic Audit Incident and Artifact Log

This document records the complete chronological inventory of forensic incidents, parsing artifacts, procedural retries, stream buffering issues, and reporting channel discrepancies documented during the audit of Paper 1 (UPAF LLPS/Heart benchmarks) and Paper 2 (EquiPhase DEQ).

| Incident ID | Target Component | Description / Discovery | Audit Classification | Resolution & Mitigation |
| :--- | :--- | :--- | :--- | :--- |
| **INC-01** | `train_paper2_deq_supervised.py` | Reported exact `0.0000e+00%` force anti-symmetry resulted from static string literal `print(f"[G1] 0.0000e+00%")`. | Hardcoded Log Literal (Pattern 2, 4th instance) | Sealed script calculation confirmed near-zero physical anti-symmetry ($0.0 \sim 2.63 \times 10^{-9}$). |
| **INC-02** | `train.tsv` (Task B) | Unicode non-breaking space (`\xa0`) caused species mnemonic misparsing for 2 non-human species strings. | Unicode Parsing Artifact | String normalization `.replace('\xa0', ' ')` introduced; 100% exact census match achieved. |
| **INC-03** | `train.tsv` (Task B) | Column named `Sequence length` contained unparseable FASTA header text rather than integer lengths. | Header-in-Length Artifact | Identified as legacy raw cell artifact (`RAWLEN`); pipeline locked to COMPLEN. |
| **INC-04** | Audit Log Maintenance | Log maintenance script (`rewrite_invalidations.py`) used write mode (`"w"`) instead of append (`"a"`), losing 5 historical entries. | Audit Log Overwrite Incident | Cryptographic hash chaining (`prev_manifest_self_sha256`) and tip anchoring introduced. |
| **INC-05** | Paper 1 §4.1 Task B | Conversation records contained strata AUROCs ($0.5898 / 0.6173$) with no anchored execution trace. | Unanchored Conversation Artifact | Sealed 16-variant audit (`v3`) proved legacy artifacts of unknown provenance; canonical block locked. |
| **INC-06** | Paper 1 §4.1 Task B | Previously blocked values ($0.6291 / 0.5756$) were reclassified as probable early unanchored calculation runs. | Reclassification of Legacy Artifacts | Formal "falsification" block label permanently revoked; UPAF principle locked. |
| **INC-07** | Task B `train.tsv` | 15 excluded rows in train split contained free text `'unknown'` in `Sequence` cell (`UNK_TEXT` policy). | Split Census Adjudication | Achieved 100% exact arithmetic census match ($n=2539$: $1734$ pos / $805$ neg). |
| **INC-08** | Paper 2 DEQ Training | Preregistration specified IFT analytical backpropagation, but actual code unrolled 100 forward solver steps. | Specification Deviation (Pattern 14) | Deviation formally documented in manuscript and `prereg_baselines_step12.md`. |
| **INC-09** | Paper 2 Preregistration | Preregistration document contained measured physical values prior to execution lock (`2cd0849`/`fc6680a`). | Post-hoc Preregistered Document (Pattern 11) | Identified git historical commits and updated document to preregistration v3. |
| **INC-10** | Paper 2 Trajectory Audit | Early audit logs contained conflict between Set A ($45/6/0/49$) and Set B ($16/37/29/18$) trajectory cross-tabulations. | Dual Output Stream Conflict | Resolved by sealed 3rd-party script (`claude_paper2_sealed_audit.py`), proving Set B authentic. |
| **INC-11** | Paper 2 Step 0-11 | Earlier audit logs claimed 100% SHA-256 hash identity between runs, traced to 108-line stream truncation. | Stream Truncation Artifact (Precursor to Pattern 13) | Full 363-line logs confirmed zero-diff bitwise reproducibility on clean output lines. |
| **INC-12** | Paper 2 Step 12 Baseline | Summary markdown text reported parameter counts `4352 / 4224` instead of raw file values `4320 / 4288`. | Invariant Parameter Canary Trigger | Raw file confirmed authentic (`4320 / 4288`); summary text classified as manual transcription error. |
| **INC-13** | Paper 2 Baseline Execution | Mid-execution read of `base_run1_raw.txt` at 16,364 bytes caused premature length mismatch report. | Output Stream Buffering Incident | Full execution wait verified 25,592 bytes complete stdout with END marker. |
| **INC-14** | Step 12 Execution Tooling | Multiple helper wrapper scripts (`py_cmp_cpu.py`, `run_cpu_baselines.py`, `run_baseline_pair.py`) generated during retries. | Wrapper Generation Incident | Wrapper scripts generation noted; direct sealed execution enforced. |
| **INC-15** | Paper 2 Step 12 Baseline | Text summary contained duplicate array hash `af1b92e8...` in two slots, 63-char hash string, and truncation markers. | Reporting Channel Reconstruction (Pattern 13) | Established reporting channel vulnerability layer; escalated to human-in-the-loop channel verification. |
| **INC-16** | Paper 2 Baseline Output | GPU execution output `base_run1_raw.txt` (hash `3C270AED...`) was overwritten during subsequent execution retries. | Evidence File Overwrite Incident | Recurrence of append/write mode file overwrite risk (§5.3); preserved in git history. |
| **INC-17** | Step 12 Output Reporting | Command lines executed but designated stdout blocks (git log, Compare-Object, `base_det1.txt` full text) were omitted. | Non-Submission of Output Stream | Occurred 3 times during baseline verification reporting; prompted channel escalation. |

---

## Pattern Taxonomy Definitions
- **Pattern 11**: Post-hoc Preregistration (preregistration documents containing measured values or git historical evidence).
- **Pattern 12**: Circular Audit Justification (justification output generating secondary discrepancies).
- **Pattern 13**: Reporting Transmission Channel Artifacts (execution is authentic, text reporting reconstructed/distorted).
- **Pattern 14**: Locked-Specification Implementation Deviation (unrolled forward solver iterations vs analytical IFT).

---

## 📌 Audit Meta-Heuristic Footnote
> *Self-reports accompanied by absolute qualifiers ("100%", "complete", "all") shall themselves be treated as re-examination triggers.*  
> *(절대 수식어('100%', '완전', '전수')가 동반된 자기 보고는 그 자체로 재검 트리거로 취급한다.)*

- **Repository**: `C:/Project/EquiPhase`
- **Audit Tooling**: Sealed 3rd-party independent scripts (`claude_paper1_sec41_recompute_v3.py`, `claude_paper2_sealed_audit.py`, `claude_paper2_baselines_sealed.py`).
