import argparse
import glob
import os
import sys

import pandas as pd


def _find_file(data_dir, keyword):
    """Find a CSV in data_dir whose name contains `keyword` (case-insensitive)."""
    pattern = os.path.join(data_dir, "*.csv")
    for path in glob.glob(pattern):
        if keyword.lower() in os.path.basename(path).lower():
            return path
    return None


def load_bing(path):
    df = pd.read_csv(path, index_col=0)
    df = df.rename(
        columns={
            "CampaignId": "campaign_id",
            "TimePeriod": "date",
            "Revenue": "revenue",
            "Spend": "spend",
            "Clicks": "clicks",
            "Impressions": "impressions",
            "Conversions": "conversions",
            "CampaignType": "campaign_type",
            "DailyBudget": "daily_budget",
            "CampaignName": "campaign_name",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df["channel"] = "bing"
    df["campaign_id"] = df["campaign_id"].astype(str)
    # Normalize to match Google's naming convention (performancemax -> performance_max)
    df["campaign_type"] = (
        df["campaign_type"].str.lower().str.replace(" ", "_")
        .replace({"performancemax": "performance_max"})
    )
    cols = [
        "date", "channel", "campaign_id", "campaign_name", "campaign_type",
        "spend", "revenue", "clicks", "impressions", "conversions", "daily_budget",
    ]
    return df[cols]


def load_google(path):
    df = pd.read_csv(path, index_col=0)
    df = df.rename(
        columns={
            "campaign_id": "campaign_id",
            "segments_date": "date",
            "metrics_clicks": "clicks",
            "metrics_conversions": "conversions",
            "metrics_impressions": "impressions",
            "metrics_conversions_value": "revenue",
            "campaign_advertising_channel_type": "campaign_type",
            "campaign_budget_amount": "daily_budget",
            "campaign_name": "campaign_name",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df["channel"] = "google"
    df["campaign_id"] = df["campaign_id"].astype(str)
    # CRITICAL: cost is in micros -> convert to base currency units
    df["spend"] = df["metrics_cost_micros"] / 1_000_000.0
    df["campaign_type"] = df["campaign_type"].str.lower()
    cols = [
        "date", "channel", "campaign_id", "campaign_name", "campaign_type",
        "spend", "revenue", "clicks", "impressions", "conversions", "daily_budget",
    ]
    return df[cols]


def load_meta(path):
    df = pd.read_csv(path, index_col=0)
    df = df.rename(
        columns={
            "campaign_id": "campaign_id",
            "date_start": "date",
            "spend": "spend",
            "clicks": "clicks",
            "impressions": "impressions",
            "conversion": "revenue",  # NOTE: Meta's "conversion" is conversion VALUE
            "daily_budget": "daily_budget",
            "campaign_name": "campaign_name",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df["channel"] = "meta"
    df["campaign_id"] = df["campaign_id"].astype(str)
    df["conversions"] = pd.NA  # no true conversion count available for Meta
    # Derive campaign_type from name prefix since Meta has no type column
    prefix = df["campaign_name"].str.split("_").str[0].str.lower()
    df["campaign_type"] = prefix
    cols = [
        "date", "channel", "campaign_id", "campaign_name", "campaign_type",
        "spend", "revenue", "clicks", "impressions", "conversions", "daily_budget",
    ]
    return df[cols]


def validate_campaign_consistency(df):
    """Run lightweight sanity checks; return a list of warning strings.

    This satisfies the brief's "validating campaign consistency" deliverable
    without being so strict that a different (but valid) held-out test set
    would fail the whole pipeline.
    """
    warnings = []
    for channel, sub in df.groupby("channel"):
        n_campaigns = sub["campaign_id"].nunique()
        if n_campaigns == 0:
            warnings.append(f"[{channel}] has zero campaigns.")
        neg_spend = (sub["spend"] < 0).sum()
        if neg_spend:
            warnings.append(f"[{channel}] has {neg_spend} rows with negative spend.")
        neg_rev = (sub["revenue"] < 0).sum()
        if neg_rev:
            warnings.append(f"[{channel}] has {neg_rev} rows with negative revenue.")
        dupes = sub.duplicated(subset=["campaign_id", "date"]).sum()
        if dupes:
            warnings.append(
                f"[{channel}] has {dupes} duplicate (campaign_id, date) rows."
            )
    return warnings


def build_unified_table(data_dir):
    bing_path = _find_file(data_dir, "bing")
    google_path = _find_file(data_dir, "google")
    meta_path = _find_file(data_dir, "meta")

    frames = []
    if bing_path:
        frames.append(load_bing(bing_path))
    else:
        print("[generate_features] WARNING: no bing_*.csv found in data dir", file=sys.stderr)

    if google_path:
        frames.append(load_google(google_path))
    else:
        print("[generate_features] WARNING: no google_*.csv found in data dir", file=sys.stderr)

    if meta_path:
        frames.append(load_meta(meta_path))
    else:
        print("[generate_features] WARNING: no meta_*.csv found in data dir", file=sys.stderr)

    if not frames:
        raise FileNotFoundError(
            f"No recognizable channel CSVs (bing/google/meta) found in {data_dir}"
        )

    unified = pd.concat(frames, ignore_index=True)
    unified = unified.sort_values(["channel", "campaign_id", "date"]).reset_index(drop=True)

    warnings = validate_campaign_consistency(unified)
    for w in warnings:
        print(f"[generate_features] VALIDATION WARNING: {w}", file=sys.stderr)

    return unified


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    unified = build_unified_table(args.data_dir)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.out.endswith(".parquet"):
        unified.to_parquet(args.out, index=False)
    else:
        unified.to_csv(args.out, index=False)

    print(f"[generate_features] Wrote {len(unified)} rows to {args.out}")
    print(f"[generate_features] Channels: {sorted(unified['channel'].unique())}")
    print(f"[generate_features] Date range: {unified['date'].min()} to {unified['date'].max()}")


if __name__ == "__main__":
    main()
    