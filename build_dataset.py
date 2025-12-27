import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_PATH = HERE / "data_spy_features.csv"


def build_dataset(ticker="SPY", start="2006-01-01"):
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)

    df = df[["Close"]].rename(columns={"Close": "close"})

    df["ret1"] = df["close"].pct_change()
    df["ret1_next"] = df["ret1"].shift(-1)
    df["y"] = (df["ret1_next"] > 0).astype(int)

    df["ret1_lag1"] = df["ret1"].shift(1)
    df["ret5"] = df["close"].pct_change(5)
    df["vol10"] = df["ret1"].rolling(10).std()
    df["vol20"] = df["ret1"].rolling(20).std()

    df = df.dropna().copy()
    df = df.reset_index()  # makes 'date' a column

    return df


if __name__ == "__main__":
    df = build_dataset()

    print(df.head())
    print("\nRows:", len(df))
    print("Start:", df["Date"].min(), "End:", df["Date"].max())
    print("\nClass balance (y=1):", df["y"].mean())

    df.to_csv(OUT_PATH, index=False)
    print("\nSaved:", OUT_PATH)
