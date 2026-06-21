# UPAF-General Task Checklist

- [x] Fix the permutation alignment bug in `upaf.py`
- [x] Run Phase 2 calibration benchmark (`run_phase2_calibration.py`)
- [x] Verify monotonic relationship of Placebo AUROC vs leakage strength $\alpha$
- [x] Design and implement Phase 3 cross-validation (`run_phase3_cross_validation.py`)
  - [x] Implement Task A: Fold-switch margin prediction (using `data/benchmark_pairs.csv`)
  - [x] Implement Task B: LLPS status prediction (using biological LLPSDB v2.0 dataset)
  - [x] Implement Task C: Sequence length prediction (clean dataset, ESM-2 residue embeddings)
- [x] Run Phase 3 cross-validation script
- [x] Verify that Task C yields `SIGNAL-GENUINE` (specificity proof)
- [x] Design and implement plotting script (`generate_plots.py`)
- [x] Check Phase 0 principles: per-seed raw data, average arithmetic formulas, non-zero placebo durations
- [/] Create/update final reports and walkthrough artifacts (`walkthrough.md` and `walkthrough_ko.md`)
