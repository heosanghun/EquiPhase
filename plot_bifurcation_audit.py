import json
import matplotlib.pyplot as plt
import numpy as np

def render_plots():
    # Load sweep and audit results
    with open("masterpiece_results.json", "r") as f:
        data = json.load(f)
        
    lams = np.array(data["sweep_lams"])
    margins = np.array(data["sweep_margins"])
    dist_A = np.array(data["sweep_dist_A"])
    dist_B = np.array(data["sweep_dist_B"])
    res_x = np.array(data["residuals_x"])
    res_y = np.array(data["residuals_y"])
    r_val = data["r_val"]
    p_val = data["p_val"]
    verdict = data["verdict"]
    
    # Configure publication quality parameters
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.unicode_minus"] = False
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Left Panel: Bifurcation Diagram & Structural Transition
    color_A = "#1f77b4" # Deep Blue
    color_B = "#ff7f0e" # Vibrant Orange
    
    # Plot stability margins (left y-axis)
    ax1.plot(lams, margins[:, 0], label="Margin $m_A$ (Fold A)", color=color_A, linewidth=2.5)
    ax1.plot(lams, margins[:, 1], label="Margin $m_B$ (Fold B)", color=color_B, linewidth=2.5)
    ax1.set_xlabel("Control Parameter $\lambda$", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Jacobian Stability Margin $m = 1 - \\rho(J)$", fontsize=12, fontweight="bold")
    ax1.tick_params(axis="y")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Create twin axis for distances (right y-axis)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(lams, dist_A, label="Dist to Fold A (Å)", color=color_A, linestyle="--", alpha=0.7, linewidth=2.0)
    ax1_twin.plot(lams, dist_B, label="Dist to Fold B (Å)", color=color_B, linestyle="--", alpha=0.7, linewidth=2.0)
    ax1_twin.set_ylabel("Coordinate RMSD to Target (Å)", fontsize=12, fontweight="bold")
    ax1_twin.tick_params(axis="y")
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=True, shadow=True)
    
    ax1.set_title("Symplectic DEQ Bifurcation Map", y=1.18, fontsize=13, fontweight="bold")
    
    # 2. Right Panel: Partial Correlation Residuals Scatter Plot
    ax2.scatter(res_x, res_y, color="#2ca02c", alpha=0.6, edgecolors="none", s=50, label="Audited Residue Samples")
    
    # Fit regression line
    m_slope, c_intercept = np.polyfit(res_x, res_y, 1)
    x_range = np.linspace(res_x.min(), res_x.max(), 100)
    ax2.plot(x_range, m_slope * x_range + c_intercept, color="#d62728", linestyle="-", linewidth=2.0, label="Fitted Regression")
    
    ax2.set_xlabel("Residuals of Stability Margin ($e_{m \\cdot \\text{Seq}}$)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Residuals of Structure Quality ($e_{\\text{RMSD} \\cdot \\text{Seq}}$)", fontsize=12, fontweight="bold")
    ax2.set_title("Physical Turing Test Residuals Audit", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    # Add text stats box
    stats_text = f"Partial Correlation r: {r_val:.4f}\np-value: {p_val:.4e}\nVerdict: {verdict}"
    props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray")
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=10, verticalalignment="top", bbox=props)
    ax2.legend(loc="lower right")
    
    plt.tight_layout()
    
    # Save files
    plt.savefig("bifurcation_audit_plot.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig("bifurcation_audit_plot.png", format="png", dpi=300, bbox_inches="tight")
    
    print("Plots successfully rendered and saved to bifurcation_audit_plot.pdf and bifurcation_audit_plot.png.")

if __name__ == "__main__":
    render_plots()
