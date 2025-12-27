import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Paths (Step 0)
# -----------------------------
HERE = Path(__file__).resolve().parent
IN_PATH = HERE / "data_spy_features.csv"
OUT_PATH = HERE / "data_spy_features_regimes.csv"


def add_gmm_regimes(df, features, n_regimes=4):
    """
    Fit a GMM on TRAIN data only and assign regimes to all rows.
    """

    X = df[features].values

    # Time-based split (no leakage)
    split = int(len(df) * 0.7)
    X_train = X[:split]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_all_scaled = scaler.transform(X)

    gmm = GaussianMixture(
        n_components=n_regimes,
        covariance_type="full",
        random_state=42
    )

    gmm.fit(X_train_scaled)

    regimes = gmm.predict(X_all_scaled)

    return regimes


if __name__ == "__main__":

    # -----------------------------
    # Load dataset
    # -----------------------------
    df = pd.read_csv(IN_PATH)

    # Ensure date column exists and is sorted
    if "Date" not in df.columns:
        raise ValueError("Expected a 'Date' column from build_dataset.py")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # -----------------------------
    # Regime features
    # -----------------------------
    regime_features = ["ret5", "vol10", "vol20"]

    # Drop rows with missing regime features
    df = df.dropna(subset=regime_features).copy()

    # -----------------------------
    # Add dynamic regimes
    # -----------------------------
    df["regime"] = add_gmm_regimes(
        df,
        features=regime_features,
        n_regimes=4
    )

    # -----------------------------
    # Diagnostics
    # -----------------------------
    print("\nRegime counts:")
    print(df["regime"].value_counts().sort_index())

    print("\nMean features by regime:")
    print(df.groupby("regime")[regime_features].mean())

    # -----------------------------
    # Save
    # -----------------------------
    df.to_csv(OUT_PATH, index=False)
    print("\nSaved:", OUT_PATH)
