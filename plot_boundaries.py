import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data_spy_features_regimes.csv"

df = pd.read_csv(DATA_PATH)

if "Date" not in df.columns:
    raise ValueError("Expected 'Date' column")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

features = ["ret5", "vol20"]
df = df.dropna(subset=features + ["y", "regime"]).copy()

X = df[features].values
y = df["y"].values
regime = df["regime"].values

# Train global model (for visualization)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000))
])

pipe.fit(X, y)

# grid for decision surface
x0 = df["ret5"].values
x1 = df["vol20"].values

x0_min, x0_max = np.percentile(x0, [1, 99])
x1_min, x1_max = np.percentile(x1, [1, 99])

xx0, xx1 = np.meshgrid(
    np.linspace(x0_min, x0_max, 300),
    np.linspace(x1_min, x1_max, 300)
)

grid = np.c_[xx0.ravel(), xx1.ravel()]
p_grid = pipe.predict_proba(grid)[:, 1].reshape(xx0.shape)


fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

# Global probability field
im = ax.contourf(
    xx0, xx1, p_grid,
    levels=25,
    alpha=0.35,
    cmap="RdBu_r"
)

# Scatter points colored by regime
scatter = ax.scatter(
    df["ret5"],
    df["vol20"],
    c=regime,
    cmap="tab10",
    s=8,
    alpha=0.35
)

# Decision boundary (p = 0.5)
ax.contour(
    xx0, xx1, p_grid,
    levels=[0.5],
    colors="black",
    linewidths=2
)

ax.set_title("Global Logistic Decision Boundary with GMM Regimes")
ax.set_xlabel("ret5 (5-day return)")
ax.set_ylabel("vol20 (20-day volatility)")

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("P(next-day return > 0)")

legend = ax.legend(
    *scatter.legend_elements(),
    title="Regime",
    loc="upper right",
    frameon=True
)

plt.tight_layout()
plt.show()
