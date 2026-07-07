import json
import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "llama-3.1-8b-instant"


def build_summary_for_prompt(predictions_df):
    total = predictions_df[predictions_df["level"] == "total"].iloc[0]
    channels = predictions_df[predictions_df["level"] == "channel"]
    ch_lines = []
    for _, r in channels.iterrows():
        ch_lines.append(
            f"{r['channel']}: revenue=${r['revenue_p50']:,.0f}, ROAS={r['roas_p50']:.2f}x"
        )
    return (
        f"Forecast ({int(total['horizon_days'])} days): "
        f"Total revenue likely ${total['revenue_p50']:,.0f} "
        f"(range ${total['revenue_p10']:,.0f}-${total['revenue_p90']:,.0f}), "
        f"ROAS {total['roas_p50']:.2f}x, spend ${total['spend_total']:,.0f}. "
        f"Channels: {'; '.join(ch_lines)}"
    )


def generate_causal_summary(predictions_df):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "[llm_insights] No GROQ_API_KEY found in environment. Check your .env file."
    
    client = Groq(api_key=api_key)
    summary = build_summary_for_prompt(predictions_df)
    
    prompt = (
        f"You are a marketing analyst. Given this ecommerce forecast: {summary} "
        f"Write 3 sentences: 1) overall outlook, 2) which channel needs attention, "
        f"3) one concrete recommendation for the marketing team."
    )
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[llm_insights] Groq API error: {e}"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.predictions)
    summary = generate_causal_summary(df)
    print("\n" + "=" * 60)
    print("AI-GENERATED CAUSAL SUMMARY")
    print("=" * 60)
    print(summary)
