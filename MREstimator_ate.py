import pandas as pd
import matplotlib.pyplot as plt
import torch

from ml_mr.estimation import quantile_iv  # Adjust path if needed

# === Load Simulated Data ===
df = pd.read_csv("simulated_data/merged_mv.csv")

# === Define True Slopes Used in Simulation ===
true_slopes = {
    'Y1': 0.6,
    'Y2': -0.2,
    'Y3': 0.1
}

# === Load Trained Quantile IV Estimator ===
estimator = quantile_iv.load("quantile_iv_output_mv")

# === Compute ATE Using .ate() Method ===
x0 = torch.tensor([[0.0]])  # Treatment baseline
x1 = torch.tensor([[1.0]])  # Treatment effect

ate = estimator.ate(x0, x1)  # Tensor shape: (1, outcome_dim)

print("🔬 Estimated causal effects using Quantile IV (ATE from MREstimator.ate):")
for i, val in enumerate(ate[0]):
    y_label = f"Y{i+1}"
    print(f"  {y_label} ~ T1: Estimated ATE = {val.item():.3f}, True slope ≈ {true_slopes[y_label]:.3f}")

# === Optional: Plot Estimated do(Y) Functions ===
df_est = pd.read_csv("quantile_iv_output_mv/causal_estimates.csv")

plt.figure()
plt.plot(df_est['x'], df_est['y_do_x_1'], '.', label='Y1 (do(X))')
plt.plot(df_est['x'], df_est['y_do_x_2'], '.', label='Y2 (do(X))')
plt.plot(df_est['x'], df_est['y_do_x_3'], '.', label='Y3 (do(X))')
plt.xlabel('X (Exposure)')
plt.ylabel('do(Y)')
plt.title('Estimated do(Y) ~ X from Quantile IV')
plt.legend()
plt.grid(True)
plt.show()
