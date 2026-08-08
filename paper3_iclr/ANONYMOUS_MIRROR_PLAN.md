# Anonymous Repository Mirror Plan (ICLR 2027 Double-Blind Submission)

To comply with ICLR 2027 double-blind review policies, all cryptographic audit trails, sealed scripts, and evidence logs will be hosted on an anonymous repository mirror (`anonymous.4open.science` or an anonymous GitHub organization) prior to submission.

## 1. Replication Target Assets
The anonymous mirror will contain the complete cryptographic proof chain:

- **Sealed Evaluation Scripts**:
  - `claude_ala2_phase1_eda_sealed.py` (Phase 1 EDA)
  - `claude_ala2_phase2_train_gates_sealed.py` (Phase 2 EquiPhase & Baselines)
  - `claude_ala2_monotone_r4a_sealed_v2.py` (Monotone v2 Banach Contraction Re-Test)
  - `claude_paper2_sealed_audit.py` (Synthetic Double-Well EquiPhase Audit)
  - `claude_paper2_baselines_sealed.py` (Synthetic Double-Well Control Baselines)

- **Cryptographic Manifests & Anchors**:
  - `PAPER3_MANIFEST_20260808.txt` (ICLR Manuscript SHA-256 Manifest)
  - `SEALED_MANIFEST_20260808.txt` (Auditor Script & Anchor Manifest)
  - `z_inits_sealed.pt` (Sealed Initialization Anchors)

- **Untruncated Evidence Logs**:
  - `evidence_p2gates_20260808_172910.txt` (P2GATES Output Log)
  - `evidence_m2_20260808_173543.txt` (M2RUN Output Log)

- **Preregistration & Audit Specifications**:
  - `PREREG_ALA2_EQUIPHASE_v1.md` (Alanine Dipeptide Preregistration v1 & v1.1)
  - `FREEZE_PAPER2.md` (Synthetic Double-Well Preregistration & Freeze State)
  - `audit_incident_log.md` (Anonymized Forensic Incident Log INC-01 through INC-22)

## 2. Anonymization Protocol
Prior to pushing to the anonymous mirror platform:
1. Run automated path scrubber to convert all Windows local drive paths (`C:\Project\EquiPhase\...`) into repository-relative paths (`./...`).
2. Strip explicit author name (`Sanghoon Huh`), GitHub handles, and lab identifiers from text logs and TeX author fields.
3. Validate bitwise SHA-256 hash preservation of all python scripts and data arrays.

## 3. Mirroring Execution
Execution will occur upon explicit approval from author (허상훈 님) prior to paper submission.
