import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Monkey-patch platform module to bypass Windows WMI query hangs
import platform
from collections import namedtuple
UnameResult = namedtuple('UnameResult', ['system', 'node', 'release', 'version', 'machine', 'processor'])
platform.win32_ver = lambda *args, **kwargs: ('10', '10.0.0', '', 'Multiprocessor Free')
platform.uname = lambda: UnameResult('Windows', 'DESKTOP-XXX', '10', '10.0.0', 'AMD64', 'AMD64')
platform.machine = lambda: 'AMD64'
platform.system = lambda: 'Windows'
platform.processor = lambda: 'AMD64'
platform.release = lambda: '10'
platform.version = lambda: '10.0.0'

class DoubleWellDEQCell(nn.Module):
    """
    DEQ Cell mimicking gradient descent dynamics on a Double-Well Potential:
    V(z, lam) = 0.25 * z^4 - 0.5 * z^2 - lam * z
    """
    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, z, lam):
        # Grad V(z, lam) = z^3 - z - lam
        grad = z**3 - z - lam
        return z - self.alpha * grad

def find_roots_analytical(lam_val):
    """
    Finds all real roots of the cubic equation: z^3 - z - lam = 0
    using numpy.roots for guaranteed convergence of both stable and unstable fixed points.
    """
    # coefficients: 1*z^3 + 0*z^2 - 1*z - lam = 0
    coeffs = [1.0, 0.0, -1.0, -float(lam_val)]
    roots = np.roots(coeffs)
    # Filter for real roots
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-5:
            real_roots.append(r.real)
    return sorted(real_roots)

def compute_jacobian_and_margin(cell, z_val, lam_val):
    """
    Computes the cell Jacobian d(f(z, lam))/dz using PyTorch Autograd,
    and returns the stability margin m = 1 - |J|.
    """
    z_tensor = torch.tensor([z_val], dtype=torch.float32, requires_grad=True)
    lam_tensor = torch.tensor([lam_val], dtype=torch.float32)
    
    z_next = cell(z_tensor, lam_tensor)
    
    # Compute Jacobian J = d(z_next)/d(z) via autograd
    J = torch.autograd.grad(z_next, z_tensor, torch.ones_like(z_next), retain_graph=True)[0]
    
    # Spectral radius for 1D scalar is simply absolute value of J
    rho_J = torch.abs(J).item()
    margin = 1.0 - rho_J
    return J.item(), margin

def main():
    print("Initializing ISS Phase 1 PoC...")
    
    # Setup directories
    output_dir = "D:/AI/EquiPhase"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "iss_bifurcation_diagram.png")
    
    # Hyperparameters
    alpha = 0.05
    cell = DoubleWellDEQCell(alpha=alpha)
    
    # Range of control parameter lambda
    lam_sweep = np.linspace(-0.6, 0.6, 300)
    
    # Storage for plotting
    stable_branch_1 = [] # Lower stable well (z < 0 at lam=0)
    stable_branch_2 = [] # Upper stable well (z > 0 at lam=0)
    unstable_branch = [] # Unstable barrier (z approx 0 at lam=0)
    
    # Sweep lambda
    for lam in lam_sweep:
        roots = find_roots_analytical(lam)
        
        # Calculate Jacobian & margin for each root
        for r in roots:
            J, m = compute_jacobian_and_margin(cell, r, lam)
            is_stable = (abs(J) < 1.0)
            
            entry = {"lam": lam, "z": r, "J": J, "margin": m, "is_stable": is_stable}
            
            # Classify branches based on stability and position
            if is_stable:
                if r < 0:
                    stable_branch_1.append(entry)
                else:
                    stable_branch_2.append(entry)
            else:
                unstable_branch.append(entry)
                
    # Convert lists to NumPy arrays for easy plotting
    sb1_lam = np.array([e["lam"] for e in stable_branch_1])
    sb1_z   = np.array([e["z"] for e in stable_branch_1])
    sb1_m   = np.array([e["margin"] for e in stable_branch_1])
    
    sb2_lam = np.array([e["lam"] for e in stable_branch_2])
    sb2_z   = np.array([e["z"] for e in stable_branch_2])
    sb2_m   = np.array([e["margin"] for e in stable_branch_2])
    
    ub_lam  = np.array([e["lam"] for e in unstable_branch])
    ub_z    = np.array([e["z"] for e in unstable_branch])
    ub_m    = np.array([e["margin"] for e in unstable_branch])
    
    # Critical Bifurcation values (Analytical)
    # lam_c = 2 / (3 * sqrt(3)) approx 0.3849
    # z_c = 1 / sqrt(3) approx 0.5774
    lam_c = 2.0 / (3.0 * np.sqrt(3.0))
    z_c = 1.0 / np.sqrt(3.0)
    
    print(f"Analytical Critical Bifurcation Parameter: lambda_c = +/- {lam_c:.4f}")
    print(f"Analytical Critical State: z_c = +/- {z_c:.4f}")
    
    # Print sample data to console to verify
    print("\n--- Sample Values Sweep ---")
    print(f"{'Lambda':<10}{'State (z*)':<15}{'Jacobian J':<15}{'Stability Margin m':<20}{'Status':<10}")
    print("-" * 75)
    for lam in [-0.5, -0.3849, 0.0, 0.3849, 0.5]:
        roots = find_roots_analytical(lam)
        for r in roots:
            J, m = compute_jacobian_and_margin(cell, r, lam)
            status = "Stable" if abs(J) < 1.0 else "Unstable"
            print(f"{lam:<10.4f}{r:<15.4f}{J:<15.4f}{m:<20.4f}{status:<10}")
            
    # Set up matplotlib style for premium aesthetics
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#CCCCCC"
    plt.rcParams["axes.linewidth"] = 0.8
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    fig.suptitle("Implicit Stability Spectroscopy (ISS) - Phase 1 PoC\nBifurcation and Stability Collapse in Double-Well Potential", fontsize=14, fontweight="bold", y=0.98)
    
    # ---------------- Left Panel: Bifurcation Diagram ----------------
    # Plot stable wells
    ax1.plot(sb1_lam, sb1_z, label="Stable Well 1 ($z^* < 0$)", color="#1F77B4", linewidth=2.5)
    ax1.plot(sb2_lam, sb2_z, label="Stable Well 2 ($z^* > 0$)", color="#2CA02C", linewidth=2.5)
    
    # Plot unstable barrier
    ax1.plot(ub_lam, ub_z, label="Unstable Barrier", color="#D62728", linestyle="--", linewidth=2.0)
    
    # Mark bifurcation points
    ax1.scatter([lam_c, -lam_c], [-z_c, z_c], color="black", edgecolor="white", s=80, zorder=5, label="Bifurcation Points")
    
    # Annotate bifurcation points
    ax1.annotate(r"$\lambda_c \approx 0.385$", xy=(lam_c, -z_c), xytext=(lam_c + 0.05, -z_c - 0.3),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6))
    ax1.annotate(r"$-\lambda_c \approx -0.385$", xy=(-lam_c, z_c), xytext=(-lam_c - 0.25, z_c + 0.3),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6))
    
    ax1.set_title(r"Bifurcation Diagram ($z^*$ vs. $\lambda$)", fontsize=11, fontweight="semibold")
    ax1.set_xlabel(r"Control Parameter $\lambda$", fontsize=10)
    ax1.set_ylabel(r"Equilibrium Fixed Point $z^*$", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="best", frameon=True, framealpha=0.9, edgecolor="#E5E5E5")
    ax1.set_xlim(-0.6, 0.6)
    ax1.set_ylim(-1.6, 1.6)
    
    # ---------------- Right Panel: Stability Margin ----------------
    # Plot stability margin m = 1 - |J|
    ax2.plot(sb1_lam, sb1_m, color="#1F77B4", linewidth=2.5, label="Stable Well 1")
    ax2.plot(sb2_lam, sb2_m, color="#2CA02C", linewidth=2.5, label="Stable Well 2")
    ax2.plot(ub_lam, ub_m, color="#D62728", linestyle="--", linewidth=2.0, label="Unstable Barrier")
    
    # Horizontal line showing instability boundary (m = 0)
    ax2.axhline(0.0, color="gray", linestyle="-.", linewidth=1)
    
    # Mark bifurcation points where m = 0
    ax2.scatter([lam_c, -lam_c], [0.0, 0.0], color="black", edgecolor="white", s=80, zorder=5)
    
    ax2.set_title(r"Stability Margin ($m = 1 - \rho(J)$)", fontsize=11, fontweight="semibold")
    ax2.set_xlabel(r"Control Parameter $\lambda$", fontsize=10)
    ax2.set_ylabel(r"Stability Margin $m$", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="best", frameon=True, framealpha=0.9, edgecolor="#E5E5E5")
    ax2.set_xlim(-0.6, 0.6)
    ax2.set_ylim(-0.15, 0.15)
    
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"\nSuccess! Plots saved to {plot_path}")

if __name__ == "__main__":
    main()
