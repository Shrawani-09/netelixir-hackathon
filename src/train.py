


import argparse
import os
import pickle

import numpy as np
import pandas as pd
from prophet import Prophet


def prepare_channel_series(df: pd.DataFrame, channel: str) -> pd.DataFrame:
    """Aggregate a channel's data to one row per day: total revenue + spend."""
    sub = df[df["channel"] == channel]
    daily = (
        sub.groupby("date")
        .agg(revenue=("revenue", "sum"), spend=("spend", "sum"))
        .reset_index()
        .sort_values("date")
    )
    # Prophet requires columns named exactly 'ds' (date) and 'y' (target)
    daily = daily.rename(columns={"date": "ds", "revenue": "y"})
    # log1p avoids log(0) issues on no-spend days
    daily["log_spend"] = np.log1p(daily["spend"])
    return daily


def fit_channel_model(daily: pd.DataFrame) -> tuple[Prophet, np.ndarray, float]:
    """Fit Prophet with log_spend as a regressor; return model + residuals."""
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",  # spikes scale with the trend, not additive
        interval_width=0.8,
    )
    model.add_regressor("log_spend")
    model.fit(daily[["ds", "y", "log_spend"]])

    # In-sample predictions -> residuals, which we'll bootstrap from later
    in_sample = model.predict(daily[["ds", "log_spend"]])
    residuals = daily["y"].values - in_sample["yhat"].values

    avg_log_spend = daily["log_spend"].mean()
    return model, residuals, avg_log_spend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Path to features.parquet")
    parser.add_argument("--out", required=True, help="Path to write pickle/model.pkl")
    args = parser.parse_args()

    df = pd.read_parquet(args.features)

    models = {}
    residuals = {}
    avg_log_spend = {}

    for channel in sorted(df["channel"].unique()):
        print(f"[train] Fitting Prophet model for channel: {channel}")
        daily = prepare_channel_series(df, channel)
        model, resid, avg_ls = fit_channel_model(daily)
        models[channel] = model
        residuals[channel] = resid
        avg_log_spend[channel] = avg_ls
        print(
            f"[train]   -> {len(daily)} days fitted, "
            f"residual std={resid.std():.2f}, mean revenue={daily['y'].mean():.2f}"
        )

    bundle = {
        "models": models,
        "residuals": residuals,
        "channel_avg_log_spend": avg_log_spend,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(bundle, f)

    print(f"[train] Saved trained bundle to {args.out}")


if __name__ == "__main__":
    main()