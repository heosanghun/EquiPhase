#!/bin/bash
set -e
set -o pipefail

echo "=== Starting Remote Execution of ICLR 2026 Masterpiece Pipeline ==="
echo "Date: $(date)"
echo "GPU Selected: $CUDA_VISIBLE_DEVICES"
nvidia-smi

echo "=== Installing Dependencies ==="
pip install torchdeq scipy transformers scikit-learn matplotlib pandas openpyxl statsmodels

echo "=== Executing Mathematical Verification ==="
python -u verify_mathematics.py 2>&1 | tee verify_math.log

echo "=== Executing UPAF Benchmark Audit ==="
python -u run_honest_audit.py 2>&1 | tee honest_run.log

echo "=== Rendering Publication Plots ==="
python -u plot_bifurcation_audit.py 2>&1 | tee plot_audit.log

echo "=== Writing Completion Summary ==="
echo "Successfully completed run!" > completion_summary.log
echo "Date: $(date)" >> completion_summary.log
cat honest_audit_report.log >> completion_summary.log

# Create a final flag to signal the monitoring loop
touch training_completed.txt
echo "=== Done ==="
