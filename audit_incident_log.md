# UPAF Cryptographic Audit Incident and Artifact Log

This document records the chronological inventory of forensic incidents, parsing artifacts, text reconstruction discrepancies, and audit reclassifications documented during the audit of Paper 1 (UPAF LLPS/Heart benchmarks) and Paper 2 (EquiPhase DEQ).

| Incident ID | Target Component | Description / Discovery | Audit Classification | Resolution & Mitigation |
| :--- | :--- | :--- | :--- | :--- |
| **INC-01** | `train_paper2_deq_supervised.py` | Reported exact `0.0000e+00%` force anti-symmetry resulted from static string literal `print(f"[G1] 0.0000e+00%")`. | Hardcoded Log Literal (Pattern 2) | Sealed script calculation confirmed near-zero physical anti-symmetry ($0.0 \sim 2.63 \times 10^{-9}$). |
| **INC-02** | `train.tsv` (Task B) | Unicode non-breaking space (`\xa0`) caused species mnemonic misparsing for 2 non-human species strings. | Unicode Parsing Artifact | String normalization `.replace('\xa0', ' ')` introduced; 100% exact census match achieved. |
| **INC-03** | `train.tsv` (Task B) | Column named `Sequence length` contained unparseable FASTA header text rather than integer lengths. | Header-in-Length Artifact | Identified as legacy raw cell artifact (`RAWLEN`); pipeline locked to COMPLEN. |
| **INC-04** | Audit Log Maintenance | Log maintenance script (`rewrite_invalidations.py`) used write mode (`"w"`) instead of append (`"a"`), losing 5 historical entries. | Audit Log Overwrite Incident | Cryptographic hash chaining (`prev_manifest_self_sha256`) and tip anchoring introduced. |
| **INC-05** | Paper 1 §4.1 Task B | Conversation records contained strata AUROCs ($0.5898 / 0.6173$) with no anchored execution trace. | Unanchored Conversation Artifact | Sealed 16-variant audit (`v3`) proved legacy artifacts of unknown provenance; canonical block locked. |
| **INC-06** | Paper 1 §4.1 Task B | Previously blocked values ($0.6291 / 0.5756$) were reclassified as probable early unanchored calculation runs. | Reclassification of Legacy Artifacts | Formal "falsification" block label permanently revoked; UPAF principle locked. |
| **INC-07** | Task B `train.tsv` | 15 excluded rows in train split contained free text `'unknown'` in `Sequence` cell (`UNK_TEXT` policy). | Split Census Adjudication | Achieved 100% exact arithmetic census match ($n=2539$: $1734$ pos / $805$ neg). |
| **INC-08** | Paper 2 Step 0-11 | Earlier audit logs claimed 100% SHA-256 hash identity between runs, traced to 108-line stream truncation. | Stream Truncation Artifact (Pattern 12) | Full 363-line logs confirmed zero-diff bitwise reproducibility on clean output lines. |
| **INC-09** | Paper 2 Step 12 Baseline | Summary markdown text reported parameter counts `4352 / 4224` instead of raw file values `4320 / 4288`. | Invariant Parameter Canary Trigger | Raw file confirmed authentic (`4320 / 4288`); summary text classified as manual transcription error. |
| **INC-10** | Paper 2 Step 12 Baseline | Text summary contained duplicate array hash `af1b92e8...` in two slots, 63-char hash string, and truncation markers. | Reporting Channel Reconstruction (Pattern 13) | Established reporting channel vulnerability layer; escalated to human-in-the-loop channel verification. |

---

## Verification & Immutable Anchoring
- **Repository**: `C:/Project/EquiPhase`
- **Audit Tooling**: Sealed 3rd-party independent scripts (`claude_paper1_sec41_recompute_v3.py`, `claude_paper2_sealed_audit.py`, `claude_paper2_baselines_sealed.py`).
