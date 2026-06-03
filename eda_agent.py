"""
Autonomous Kaggle Agent — Phase 2
EDA Agent: reads data, profiles it, outputs ML strategy
"""

import os
import json
import pandas as pd
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

DATA_DIR = r"C:\Users\abhishek\data"


def load_data() -> dict:
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")

    profile = {
        "train_shape": train.shape,
        "test_shape": test.shape,
        "columns": list(train.columns),
        "dtypes": train.dtypes.astype(str).to_dict(),
        "missing": train.isnull().sum().to_dict(),
        "missing_pct": (train.isnull().mean() * 100).round(2).to_dict(),
        "target_distribution": train["Survived"].value_counts().to_dict(),
        "numeric_stats": train.describe().round(3).to_dict(),
        "sample_rows": train.head(5).to_dict(orient="records"),
        "categorical_counts": {
            col: train[col].value_counts().head(5).to_dict()
            for col in train.select_dtypes("object").columns
        }
    }
    return profile


def run_eda_agent():
    print("\n" + "="*60)
    print("EDA AGENT — Titanic")
    print("="*60 + "\n")

    profile = load_data()
    print("[EDA] Data loaded. Sending to Claude for analysis...\n")

    prompt = f"""You are an expert Kaggle data scientist analysing the Titanic dataset.

Here is the full data profile:
{json.dumps(profile, indent=2)}

Produce a structured analysis covering:
1. Dataset overview (size, target balance, task type)
2. Missing values — which columns, severity, recommended strategy
3. Key features — which are most likely predictive and why
4. Feature engineering ideas (at least 5 specific ones)
5. Recommended model types for this dataset
6. Potential pitfalls to avoid
7. Exact ML strategy to follow (preprocessing → features → models → validation)

Be specific and actionable. This output will be used by the model building agent."""

    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )

    strategy = response.content[0].text
    print(strategy)

    # Save for Phase 3
    os.makedirs("logs", exist_ok=True)
    with open("logs/eda_strategy.json", "w") as f:
        json.dump({
            "profile": profile,
            "strategy": strategy
        }, f, indent=2)

    print("\n[Done] Strategy saved to logs/eda_strategy.json")
    return strategy


if __name__ == "__main__":
    run_eda_agent()
