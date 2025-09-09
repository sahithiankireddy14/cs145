import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Data
Pred = [0.3417, 0.2857, 0.1667, 0.3125, 0.1125, 0.1125, 0.3325, 0.2925, 0.235,
        1.0, 0.32, 0.32, 0.75, 0.25, 0.1675, 0.67, 0.4, 0.15]
G = [0.8, 0.3, 0.0, 0.85, 0.3, 0.1, 0.85, 0.65, 0.1,
     1.0, 0.3, 0.0, 1.0, 0.3, 0.0, 1.0, 0.5, 0.0]

x = np.arange(len(Pred))

# ----- Plot 1: Line plot comparing values -----
plt.figure(figsize=(10,6))
plt.plot(x, G, marker='o', color='blue', label='Ground Truth (G)')
plt.plot(x, Pred, marker='s', color='orange', label='Predicted (Pred)')
plt.xlabel("Sample Index")
plt.ylabel("Similarity Score")
plt.title("Predicted vs Ground Truth Similarity Scores")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----- Plot 2: Scatter plot with both Pearson and Spearman correlations -----
pearson_r, p_p = pearsonr(G, Pred)
spearman_rho, p_s = spearmanr(G, Pred)

plt.figure(figsize=(8,6))
plt.scatter(G, Pred, color='orange', label='Pred vs G')
sns.regplot(x=G, y=Pred, scatter=False, color='red', label='Regression Line')
plt.xlabel("Ground Truth Similarity (G)")
plt.ylabel("Predicted Similarity (Pred)")
plt.title("Scatter: Predicted vs Ground Truth\n"
          f"Pearson r = {pearson_r:.2f} (p={p_p:.4g}), "
          f"Spearman ρ = {spearman_rho:.2f} (p={p_s:.4g})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
