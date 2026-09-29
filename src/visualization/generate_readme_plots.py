"""
generate_readme_plots.py - Generate publication-ready plots for the README.

Run from project root:
    python src/visualization/generate_readme_plots.py

All plots are saved to docs/ for use in README.md.
"""

import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless saving
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Ensure docs/ folder exists
os.makedirs('docs', exist_ok=True)

print("Generating README plots...\n")

# ==========================================
# PLOT 1: Confusion Matrix
# ==========================================
print("[1/3] Confusion matrix...")
cm = np.array([[496, 4], [108, 392]])

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'])
plt.title('Confusion Matrix (Best Model: 89.1% Accuracy)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('docs/readme_confusion_matrix.png', dpi=150)
plt.close()
print("   Saved: docs/readme_confusion_matrix.png")

# ==========================================
# PLOT 2: Feature Ablation Bar Chart
# ==========================================
print("[2/3] Feature ablation chart...")
models = ['Magnitude-only', 'Phase-only', 'Fused']
means = [86.50, 75.50, 83.17]
stds = [0.35, 11.14, 4.18]

plt.figure(figsize=(8, 5))
plt.bar(models, means, yerr=stds, capsize=10,
        color=['#3498db', '#e74c3c', '#2ecc71'])
plt.ylabel('Accuracy (%)')
plt.title('Feature Ablation Study (Mean ± Std over 3 Seeds)')
plt.ylim(0, 100)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('docs/readme_ablation_bar.png', dpi=150)
plt.close()
print("   Saved: docs/readme_ablation_bar.png")

# ==========================================
# PLOT 3: Two-Branch Comparison
# ==========================================
print("[3/3] Two-branch comparison...")
models_tb = ['Magnitude-only', 'Naive Fused', 'Two-Branch']
means_tb = [88.97, 89.10, 88.90]
stds_tb = [0.12, 0.08, 0.08]

plt.figure(figsize=(8, 5))
plt.bar(models_tb, means_tb, yerr=stds_tb, capsize=10,
        color=['#3498db', '#f39c12', '#9b59b6'])
plt.ylabel('Accuracy (%)')
plt.title('Two-Branch Architecture Comparison (5K Samples, 3 Seeds)')
plt.ylim(0, 100)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('docs/readme_two_branch.png', dpi=150)
plt.close()
print("   Saved: docs/readme_two_branch.png")

print("\nAll plots saved to docs/")
