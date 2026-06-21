# Task Checklist: Contact-Gated Biophysical Force Field (CG-BFF) Implementation & Verification

## Phase 1: Local Code Refactoring (CG-BFF Implementation)
- [x] Implement pairwise sequence energy projection network in `SymplecticDEQ.__init__` in `equiphase/models/symplectic_deq.py`
- [x] Refactor `force_forward` in `equiphase/models/symplectic_deq.py` to use Gaussian RBF contact-gated forces
- [x] Verify that `verify_mathematics.py` passes successfully locally
- [x] Run syntax checks to ensure python compile success

## Phase 2: Remote Launch & Verification
- [x] Sync updated model files to the remote GPU server
- [x] Run remote mathematical verification via SSH
- [x] Launch `run_honest_audit.py` in a remote tmux session on GPU 5

## Phase 3: Monitor & Download Results
- [x] Monitor training logs
- [x] Download final audit logs and results
- [x] Verify that the Placebo model AUROC collapses to chance (~0.50) while the true Symplectic model achieves high AUROC
- [x] Update `walkthrough.md` with the final findings and G-BFF architecture validation
