import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data_spy_features_regimes.csv"   # after add_regimes.py
OUT_DIR = HERE / "paper_assets"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

split = int(len(df) * 0.7)
df_train = df.iloc[:split].copy()
df_test  = df.iloc[split:].copy()

# -----------------------
# 1) Quick dataset summary table (LaTeX)
# -----------------------
summary = pd.DataFrame({
    "Start date": [df["Date"].min().date()],
    "End date": [df["Date"].max().date()],
    "N obs": [len(df)],
    "Train obs": [len(df_train)],
    "Test obs": [len(df_test)],
    "Class balance (y=1)": [df["y"].mean()]
})

(summary.to_latex(index=False, float_format="%.4f")
 ).replace("\\toprule", "\\toprule\n\\midrule")  # minor style tweak

with open(OUT_DIR / "table_dataset_summary.tex", "w") as f:
    f.write(summary.to_latex(index=False, float_format="%.4f"))

# -----------------------
# 2) Descriptive statistics table (features + returns)
# -----------------------
cols = ["ret1", "ret1_next", "ret5", "vol10", "vol20"]
desc = df[cols].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]

with open(OUT_DIR / "table_descriptive_stats.tex", "w") as f:
    f.write(desc.to_latex(float_format="%.6f"))

# -----------------------
# 3) Correlation table (compact)
# -----------------------
corr = df[cols].corr()

with open(OUT_DIR / "table_corr.tex", "w") as f:
    f.write(corr.to_latex(float_format="%.3f"))

# -----------------------
# 4) Regime summary table
# -----------------------
if "regime" in df.columns:
    reg = (df.groupby("regime")[["ret5", "vol10", "vol20"]]
             .agg(["count", "mean", "std"]))
    with open(OUT_DIR / "table_regime_summary.tex", "w") as f:
        f.write(reg.to_latex(float_format="%.6f"))

# -----------------------
# 5) Figure: vol20 over time
# -----------------------
plt.figure(figsize=(10, 3), dpi=200)
plt.plot(df["Date"], df["vol20"])
plt.title("SPY 20-day Realized Volatility (vol20)")
plt.xlabel("Date")
plt.ylabel("vol20")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_vol20_timeseries.png")
plt.close()

# -----------------------
# 6) Figure: histogram of next-day returns
# -----------------------
plt.figure(figsize=(6, 4), dpi=200)
plt.hist(df["ret1_next"].dropna(), bins=60)
plt.title("Distribution of Next-Day Returns (ret1_next)")
plt.xlabel("ret1_next")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_ret1next_hist.png")
plt.close()

# -----------------------
# 7) Figure: regimes through time (simple scatter ribbon)
# -----------------------
if "regime" in df.columns:
    plt.figure(figsize=(10, 2.8), dpi=200)
    plt.scatter(df["Date"], df["regime"], s=3, alpha=0.6)
    plt.title("Inferred Market Regimes Over Time (GMM)")
    plt.xlabel("Date")
    plt.ylabel("Regime")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_regime_timeline.png")
    plt.close()

print("Saved tables/figures to:", OUT_DIR)
