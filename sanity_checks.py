import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("simulated_data/merged_mv.csv")

# Linear regression fits
for y_col in ['Y1', 'Y2', 'Y3']:
    slope, intercept = np.polyfit(df['T1'], df[y_col], 1)
    print(f"True causal relationship: {y_col} ≈ {slope:.3f} * T1 + {intercept:.3f}")

    # Plot
    sns.regplot(x='T1', y=y_col, data=df, line_kws={'color': 'red'})
    plt.title(f'{y_col} vs T1 (true data)')
    plt.xlabel('T1 (Exposure)')
    plt.ylabel(y_col)
    plt.grid(True)
    plt.show()


# Load results
df = pd.read_csv("quantile_iv_output_mv/causal_estimates.csv")

x_vals = df['x'].values
y1_vals = df['y_do_x_1'].values
y2_vals = df['y_do_x_2'].values
y3_vals = df['y_do_x_3'].values

# Linear fit (causal slope) for each outcome
slope1, _ = np.polyfit(x_vals, y1_vals, 1)
slope2, _ = np.polyfit(x_vals, y2_vals, 1)
slope3, _ = np.polyfit(x_vals, y3_vals, 1)

print(f"Estimated causal effect (Y1 ~ X): {slope1:.3f}")
print(f"Estimated causal effect (Y2 ~ X): {slope2:.3f}")
print(f"Estimated causal effect (Y3 ~ X): {slope3:.3f}")

# Optional: plot
plt.figure()
plt.plot(x_vals, y1_vals, '.', label='Y1')
plt.plot(x_vals, y2_vals, '.', label='Y2')
plt.plot(x_vals, y3_vals, '.', label='Y3')
plt.xlabel('X (Exposure)')
plt.ylabel('do(Y)')
plt.legend()
plt.title('Estimated do(Y) ~ do(X)')
plt.show()
