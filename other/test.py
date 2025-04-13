import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Create pseudodataset
# ----------------------------
np.random.seed(42)

# DFU: Faster responses, less skew
dfu_rt = np.random.exponential(scale=2.0, size=200) + np.random.normal(loc=0.5, scale=0.2, size=200)

# F: Slower responses, more skew
f_rt = np.random.exponential(scale=3.5, size=200) + np.random.normal(loc=0.7, scale=0.3, size=200)

# Assemble into DataFrame
df = pd.DataFrame({
    "reaction_time_TEST": np.concatenate([dfu_rt, f_rt]),
    "experimental_condition": ["DFU"] * len(dfu_rt) + ["F"] * len(f_rt)
})

# ----------------------------
# Empirical CDF Plot
# ----------------------------
for group, color in [("DFU", "tab:blue"), ("F", "tab:orange")]:
    vals = df[df["experimental_condition"] == group]["reaction_time_TEST"].dropna()
    sorted_vals = np.sort(vals)
    cdf = np.arange(1, len(vals)+1) / len(vals)
    plt.plot(sorted_vals, cdf, label=group, color=color)

plt.xlabel("Reaction Time (s)")
plt.ylabel("Cumulative Probability")
plt.legend()
plt.title("Empirical CDFs of Simulated RTs")
plt.grid(True)
plt.tight_layout()
plt.show()
