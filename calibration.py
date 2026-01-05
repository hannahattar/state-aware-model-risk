import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

df = pd.read_csv("data_spy_features_regimes.csv")


df = df.dropna(subset=["p", "y"])

y = df["y"].astype(int).values
p = df["p"].astype(float).values

# Brier score
brier = brier_score_loss(y, p)
print("Brier score:", round(brier, 6))

# Reliability diagram (calibration curve)
prob_true, prob_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")

plt.figure(figsize=(5, 5), dpi=150)
plt.plot(prob_pred, prob_true, marker="o")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("Mean predicted probability")
plt.ylabel("Empirical frequency of y=1")
plt.title("Reliability Diagram (Out-of-Sample)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 3), dpi=150)
plt.hist(p, bins=30)
plt.xlabel("Predicted probability p")
plt.ylabel("Count")
plt.title("Predicted Probability Distribution (Out-of-Sample)")
plt.tight_layout()
plt.show()
