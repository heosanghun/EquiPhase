import os
import json
import matplotlib.pyplot as plt
import numpy as np

def generate_plots():
    print("========================================================")
    print("PHASE 4: PLOTTING UPAF AUDIT DASHBOARD")
    print("========================================================\n")
    
    # 1. Load Calibration Results
    cal_path = "D:/AI/EquiPhase/data/upaf_calibration_results.json"
    with open(cal_path, "r") as f:
        cal_data = json.load(f)
        
    alphas = [x["alpha"] for x in cal_data]
    placebo_aurocs = [x["avg_placebo"] for x in cal_data]
    
    # 2. Load Cross-Validation Results
    cv_path = "D:/AI/EquiPhase/data/upaf_cross_validation_results.json"
    with open(cv_path, "r") as f:
        cv_data = json.load(f)
        
    tasks = ["Task A (Fold-Switch)", "Task B (LLPS Leak)", "Task C (Seq Len Clean)"]
    random_scores = [
        cv_data["task_a"]["avg_random"],
        cv_data["task_b"]["avg_random"],
        cv_data["task_c"]["avg_random"]
    ]
    disjoint_scores = [
        cv_data["task_a"]["avg_disjoint"],
        cv_data["task_b"]["avg_disjoint"],
        cv_data["task_c"]["avg_disjoint"]
    ]
    placebo_scores = [
        cv_data["task_a"]["avg_placebo"],
        cv_data["task_b"]["avg_placebo"],
        cv_data["task_c"]["avg_placebo"]
    ]
    
    # Create beautiful plots
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left Plot: Calibration Curve
    ax1 = axes[0]
    ax1.plot(alphas, placebo_aurocs, marker='o', linewidth=2.5, color='#e74c3c', label='Placebo AUROC')
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='Chance (0.5)')
    ax1.set_xlabel('Leakage Strength (alpha)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Placebo AUROC', fontsize=12, fontweight='bold')
    ax1.set_title('UPAF Calibration Curve (Placebo vs. alpha)', fontsize=14, fontweight='bold')
    ax1.set_xticks(alphas)
    ax1.set_ylim(0.4, 1.05)
    ax1.legend(loc='lower right', frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Right Plot: Cross-Validation Gaps
    ax2 = axes[1]
    x_indices = np.arange(len(tasks))
    bar_width = 0.25
    
    rects1 = ax2.bar(x_indices - bar_width, random_scores, bar_width, label='Random Split', color='#3498db')
    rects2 = ax2.bar(x_indices, disjoint_scores, bar_width, label='Disjoint Split', color='#2ecc71')
    rects3 = ax2.bar(x_indices + bar_width, placebo_scores, bar_width, label='Placebo Split', color='#9b59b6')
    
    ax2.set_xlabel('Evaluation Tasks', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Performance metric (AUROC for A/B, R2 for C)', fontsize=11, fontweight='bold')
    ax2.set_title('UPAF Cross-Validation Gap Analysis', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(tasks, fontweight='bold')
    ax2.legend(loc='lower left', frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # Label the heights on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax2.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
                        
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    
    # Save directly to artifacts directory
    artifact_path = "C:/Users/Sims/.gemini/antigravity/brain/e20d7f14-205f-4a52-9696-5f6f1c4caac8/upaf_audit_dashboard.png"
    plt.savefig(artifact_path, dpi=150)
    # Also save to workspace for convenience
    workspace_path = "D:/AI/EquiPhase/upaf_audit_dashboard.png"
    plt.savefig(workspace_path, dpi=150)
    
    print(f"UPAF Dashboard plots saved to:")
    print(f"  Artifact:  {artifact_path}")
    print(f"  Workspace: {workspace_path}")

if __name__ == "__main__":
    generate_plots()
