# NetElixir AIgnition 2026 — Probabilistic Revenue & ROAS Forecaster

An AI-assisted forecasting utility that predicts ecommerce revenue and ROAS
across Google Ads, Meta Ads, and Bing Ads using historical campaign data.

## Team
- Shrawani Palange — IIT Indore

## Python Version
Python 3.13.2

## How to Run Locally

### 1. Clone the repo
git clone https://github.com/Shrawani-09/netelixir-hackathon.git
cd netelixir-hackathon

### 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\Activate  # Windows
source venv/bin/activate  # Mac/Linux

### 3. Install dependencies
pip install -r requirements.txt

### 4. Add your Groq API key
Create a .env file in the root folder:
GROQ_API_KEY=your_key_here

Get a free key at https://console.groq.com

### 5. Run the forecasting pipeline
bash run.sh

Or with custom paths:
bash run.sh ./data ./pickle/model.pkl ./output/predictions.csv

### 6. Launch the Streamlit demo
streamlit run app.py

## What the Pipeline Does
1. generate_features.py — Ingests and normalizes the 3 channel CSVs into one unified table
2. predict.py — Fits Prophet models per channel, generates P10/P50/P90 revenue and ROAS forecasts using bootstrapped residuals
3. llm_insights.py — Calls Groq API (Llama 3.1) to generate plain-English causal summaries
4. app.py — Streamlit UI for interactive exploration, budget simulation and AI insights

## Forecasting Methodology
- Model: Facebook Prophet with yearly + weekly seasonality (multiplicative mode)
- Uncertainty: Bootstrap resampling of historical residuals (N=2000 simulations)
- Budget simulation: log(spend) as a Prophet regressor, capturing diminishing returns
- Aggregation: P10/P50/P90 percentiles of summed daily simulations over the forecast horizon

## Architecture
- Frontend: Streamlit
- Backend: Python (Prophet, pandas, numpy)
- AI Layer: Groq API (llama-3.1-8b-instant)
- Data: Google Ads, Meta Ads, Bing Ads campaign CSVs

## Assumptions and Limitations
- Existing channel-level attribution is treated as source of truth (no custom attribution engine)
- Meta conversion column is conversion value (revenue), not conversion count
- Forecasts are aggregate-period (30/60/90 day totals), not daily
- Budget simulation assumes log-linear spend-revenue relationship per channel
- AI insights require a valid GROQ_API_KEY in .env file
