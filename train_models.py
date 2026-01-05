import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data_spy_features_regimes.csv"

df = pd.read_csv(DATA_PATH)

if "Date" not in df.columns:
    raise ValueError("Expected 'Date' column from build_dataset.py")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Features and target
features = ["ret5", "vol20"]
target = "y"

df = df.dropna(subset=features + [target]).copy()

X = df[features].values
y = df[target].values

# train/test split
split = int(len(df) * 0.7)

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

# Train logistic
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000))
])

pipe.fit(X_train, y_train)

# Predict
df["p"] = np.nan
df["pred"] = np.nan

df.loc[split:, "p"] = pipe.predict_proba(X_test)[:, 1]
df.loc[split:, "pred"] = (df.loc[split:, "p"] >= 0.5).astype(int)

# Save updated dataset
df.to_csv(DATA_PATH, index=False)
print("Model trained and predictions saved to:", DATA_PATH)


acc = (df.loc[split:, "pred"] == df.loc[split:, "y"]).mean()
print("Test accuracy:", round(acc, 4))
print("Test sample size:", len(df) - split)
