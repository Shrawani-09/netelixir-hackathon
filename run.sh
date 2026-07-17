#!/usr/bin/env bash
set -euo pipefail

# Accept arguments with sensible defaults for local runs
DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "[run.sh] Step 1: Generating features from $DATA_DIR"
python src/generate_features.py \
    --data-dir "$DATA_DIR" \
    --out features.parquet

echo "[run.sh] Step 2: Generating probabilistic forecast"
python src/predict.py \
    --features features.parquet \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_PATH" \
    --horizon 30

echo "[run.sh] Step 3: Generating AI causal summary"
python src/llm_insights.py \
    --predictions "$OUTPUT_PATH" || echo "[run.sh] AI insights unavailable, skipping."

echo "[run.sh] Done. Predictions written to $OUTPUT_PATH"
