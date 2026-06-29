

import argparse
import os
import pickle

import numpy as np
import pandas as pd

N_BOOTSTRAP = 2000  # number of simulated future trajectories per channel
RNG_SEED = 42  # fixed seed -> reproducible results, as the submission guide requires


def get_campaign_type_mix(features_df: pd.DataFrame, channel: str) -> pd.DataFrame:
    """Historical share of spend by campaign_type within a channel, used to
    split the channel-level forecast down to campaign-type level."""
    sub = features_df[features_df["channel"] == channel]
    by_type = sub.groupby("campaign_type").agg(spend=("spend", "sum")).reset_index()
    total = by_type["spend"].sum()
    by_type["share"] = by_type["spend"] / total if total > 0 else 0
    return by_type[["campaign_type", "share"]]


def forecast_channel(
    model,
    residuals: np.ndarray,
    avg_log_spend: float,
    horizon_days: int,
    last_date: pd.Timestamp,
    future_daily_spend: float | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Returns (array of N_BOOTSTRAP total-revenue simulations, total spend
    over the horizon)."""
    future_dates = pd.date_range(
        last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D"
    )

    if future_daily_spend is not None:
        log_spend_value = np.log1p(future_daily_spend)
        total_spend = future_daily_spend * horizon_days
    else:
        log_spend_value = avg_log_spend
        total_spend = (np.expm1(avg_log_spend)) * horizon_days

    future_df = pd.DataFrame({"ds": future_dates, "log_spend": log_spend_value})
    point_forecast = model.predict(future_df)
    yhat = point_forecast["yhat"].values  # length = horizon_days

    # Bootstrap: for each of N_BOOTSTRAP simulated futures, resample a
    # residual for every day in the horizon and add it to that day's yhat.
    sims = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        sampled_residuals = rng.choice(residuals, size=horizon_days, replace=True)
        daily_sim = yhat + sampled_residuals
        daily_sim = np.clip(daily_sim, a_min=0, a_max=None)  # revenue can't be negative
        sims[i] = daily_sim.sum()

    return sims, total_spend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--horizon", type=int, default=30, choices=[30, 60, 90],
        help="Forecast horizon in days (30, 60, or 90)",
    )
    parser.add_argument(
        "--budget", type=str, default=None,
        help=(
            "Optional future daily budget override per channel, e.g. "
            "'google=500,meta=200,bing=20'. If omitted, uses historical "
            "average spend per channel."
        ),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(RNG_SEED)

    features_df = pd.read_parquet(args.features)
    with open(args.model, "rb") as f:
        bundle = pickle.load(f)

    models = bundle["models"]
    residuals = bundle["residuals"]
    avg_log_spend = bundle["channel_avg_log_spend"]

    budget_overrides = {}
    if args.budget:
        for pair in args.budget.split(","):
            ch, val = pair.split("=")
            budget_overrides[ch.strip()] = float(val)

    rows = []
    all_channel_sims = []
    all_channel_spend = 0.0

    for channel in sorted(models.keys()):
        sub = features_df[features_df["channel"] == channel]
        last_date = sub["date"].max()
        future_spend = budget_overrides.get(channel)

        sims, total_spend = forecast_channel(
            model=models[channel],
            residuals=residuals[channel],
            avg_log_spend=avg_log_spend[channel],
            horizon_days=args.horizon,
            last_date=last_date,
            future_daily_spend=future_spend,
            rng=rng,
        )

        p10, p50, p90 = np.percentile(sims, [10, 50, 90])
        roas_p10 = p10 / total_spend if total_spend > 0 else 0
        roas_p50 = p50 / total_spend if total_spend > 0 else 0
        roas_p90 = p90 / total_spend if total_spend > 0 else 0

        rows.append({
            "level": "channel",
            "channel": channel,
            "campaign_type": "all",
            "horizon_days": args.horizon,
            "revenue_p10": round(p10, 2),
            "revenue_p50": round(p50, 2),
            "revenue_p90": round(p90, 2),
            "spend_total": round(total_spend, 2),
            "roas_p10": round(roas_p10, 3),
            "roas_p50": round(roas_p50, 3),
            "roas_p90": round(roas_p90, 3),
        })

        # Split this channel's simulation down to campaign-type level using
        # each type's historical share of spend within the channel.
        type_mix = get_campaign_type_mix(features_df, channel)
        for _, type_row in type_mix.iterrows():
            share = type_row["share"]
            type_sims = sims * share
            type_spend = total_spend * share
            tp10, tp50, tp90 = np.percentile(type_sims, [10, 50, 90])
            t_roas_p50 = tp50 / type_spend if type_spend > 0 else 0
            t_roas_p10 = tp10 / type_spend if type_spend > 0 else 0
            t_roas_p90 = tp90 / type_spend if type_spend > 0 else 0
            rows.append({
                "level": "campaign_type",
                "channel": channel,
                "campaign_type": type_row["campaign_type"],
                "horizon_days": args.horizon,
                "revenue_p10": round(tp10, 2),
                "revenue_p50": round(tp50, 2),
                "revenue_p90": round(tp90, 2),
                "spend_total": round(type_spend, 2),
                "roas_p10": round(t_roas_p10, 3),
                "roas_p50": round(t_roas_p50, 3),
                "roas_p90": round(t_roas_p90, 3),
            })

        all_channel_sims.append(sims)
        all_channel_spend += total_spend

    # Aggregate (blended) total across all channels
    total_sims = np.sum(all_channel_sims, axis=0)
    tp10, tp50, tp90 = np.percentile(total_sims, [10, 50, 90])
    blended_roas_p10 = tp10 / all_channel_spend if all_channel_spend > 0 else 0
    blended_roas_p50 = tp50 / all_channel_spend if all_channel_spend > 0 else 0
    blended_roas_p90 = tp90 / all_channel_spend if all_channel_spend > 0 else 0

    rows.insert(0, {
        "level": "total",
        "channel": "all",
        "campaign_type": "all",
        "horizon_days": args.horizon,
        "revenue_p10": round(tp10, 2),
        "revenue_p50": round(tp50, 2),
        "revenue_p90": round(tp90, 2),
        "spend_total": round(all_channel_spend, 2),
        "roas_p10": round(blended_roas_p10, 3),
        "roas_p50": round(blended_roas_p50, 3),
        "roas_p90": round(blended_roas_p90, 3),
    })

    output_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output_df.to_csv(args.output, index=False)

    print(f"[predict] Wrote {len(output_df)} forecast rows to {args.output}")
    print(output_df.to_string(index=False))


if __name__ == "__main__":
    main()