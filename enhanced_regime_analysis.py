import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data_spy_features_regimes.csv"


df = pd.read_csv(DATA_PATH)

if "Date" not in df.columns:
    raise ValueError("Expected 'Date' column")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

split = int(len(df) * 0.7)
df_test = df.loc[split:].copy()

required_cols = ["p", "pred", "y", "ret1_next", "regime"]
missing = [c for c in required_cols if c not in df_test.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Confidence and correctness
df_test["confidence"] = np.abs(df_test["p"] - 0.5)
df_test["correct"] = (df_test["pred"] == df_test["y"]).astype(int)

# Confidence bins
df_test["conf_bin"] = pd.qcut(
    df_test["confidence"],
    q=10,
    duplicates="drop"
)

# Accuracy vs confidence by regime
stats = (
    df_test
    .groupby(["regime", "conf_bin"], observed=True)
    .agg(
        acc=("correct", "mean"),
        count=("correct", "size")
    )
    .reset_index()
)

print("\nAccuracy by confidence bin and regime:")
print(stats)

# Plot
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

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

print("\nConfidence summary:")
print(df_test["confidence"].describe())


# Strategy construction
# Base directional signal
df_test["position"] = np.where(df_test["p"] >= 0.5, 1.0, -1.0)

# Confidence filter
conf_threshold = 0.02
df_test["position_filt"] = np.where(
    df_test["confidence"] >= conf_threshold,
    df_test["position"],
    0.0
)

# Turn strategy off in 3
df_test.loc[df_test["regime"] == 3, "position_filt"] = 0.0


# Strategy returns
df_test["strategy_ret"] = df_test["position_filt"] * df_test["ret1_next"]

# Performance metrics
def sharpe(returns, rf=0.05, periods=252):
    r = returns.dropna()
    if r.std() == 0:
        return np.nan
    return np.sqrt(periods) * (r.mean() - rf / periods) / r.std()

overall_sharpe = sharpe(df_test["strategy_ret"])
print("\nOverall Sharpe:", round(overall_sharpe, 3))

# Equity curve and drawdown
df_test["equity"] = (1 + df_test["strategy_ret"]).cumprod()
df_test["peak"] = df_test["equity"].cummax()
df_test["drawdown"] = df_test["equity"] / df_test["peak"] - 1

print("Max drawdown:", round(df_test["drawdown"].min(), 3))
print("Trade frequency:", (df_test["position_filt"] != 0).mean())

# Sharpe by regime
regime_sharpes = (
    df_test
    .groupby("regime")
    .apply(lambda x: sharpe(x["strategy_ret"]))
    .sort_values(ascending=False)
)

print("\nSharpe by regime:")
print(regime_sharpes)

# Plot
plt.figure(figsize=(10, 4), dpi=150)
plt.plot(df_test["Date"], df_test["equity"])
plt.title("Strategy Equity Curve (Out-of-Sample)")
plt.xlabel("Date")
plt.ylabel("Equity")
plt.tight_layout()
plt.show()
