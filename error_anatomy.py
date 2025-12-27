import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Paths (Step 0)
# -----------------------------
HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data_spy_features_regimes.csv"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)

if "Date" not in df.columns:
    raise ValueError("Expected 'Date' column")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# -----------------------------
# Use test set only
# -----------------------------
split = int(len(df) * 0.7)
df_test = df.loc[split:].copy()

# Ensure required columns exist
required = ["p", "pred", "y", "regime"]
missing = [c for c in required if c not in df_test.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# -----------------------------
# Error + confidence
# -----------------------------
df_test["confidence"] = np.abs(df_test["p"] - 0.5)
df_test["correct"] = (df_test["pred"] == df_test["y"]).astype(int)
df_test["error"] = 1 - df_test["correct"]

# -----------------------------
# Confidence bins
# -----------------------------
df_test["conf_bin"] = pd.qcut(
    df_test["confidence"],
    q=10,
    duplicates="drop"
)

# -----------------------------
# Aggregate stats by regime
# -----------------------------
stats = (
    df_test
    .groupby(["regime", "conf_bin"], observed=True)
    .agg(
        acc=("correct", "mean"),
        err_rate=("error", "mean"),
        count=("error", "size")
    )
    .reset_index()
)

print("\nError anatomy by regime and confidence bin:")
print(stats)

# -----------------------------
# Plot accuracy vs confidence
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

for r, sub in stats.groupby("regime"):
    ax.plot(
        sub["conf_bin"].astype(str),
        sub["acc"],
        marker="o",
        label=f"Regime {r}"
    )

ax.set_title("Accuracy vs Confidence by Regime (Out-of-Sample)")
ax.set_xlabel("Confidence decile |p − 0.5|")
ax.set_ylabel("Accuracy")
ax.legend(frameon=True)
ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()
