import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- PLOT 1: Confusion Matrix ---
cm = np.array([[496, 4], [108, 392]]) # From your two_branch_results.json

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Real', 'Fake'], 
            yticklabels=['Real', 'Fake'])
plt.title('Confusion Matrix (Best Model: 89.1% Accuracy)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('results/readme_confusion_matrix.png', dpi=150)
plt.close()
print("Saved confusion matrix.")

# --- PLOT 2: Ablation Study Bar Chart ---
models = ['Magnitude-only', 'Phase-only', 'Fused']
means = [86.50, 75.50, 83.17]
stds = [0.35, 11.14, 4.18]

plt.figure(figsize=(8, 5))
plt.bar(models, means, yerr=stds, capsize=10, color=['#3498db', '#e74c3c', '#2ecc71'])
plt.ylabel('Accuracy (%)')
plt.title('Feature Ablation Study (Mean ± Std over 3 Seeds)')
plt.ylim(0, 100)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/readme_ablation_bar.png', dpi=150)
plt.close()
print("Saved ablation bar chart.")