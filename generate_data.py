from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N = 6000
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "loan_default_nigeria_synthetic.csv"


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def main():
    rng = np.random.default_rng(SEED)

    age = rng.integers(18, 66, N)
    gender = rng.choice(["Female", "Male"], N, p=[0.52, 0.48])
    employment_type = rng.choice(
        ["Salaried", "Self-employed", "Trader", "Casual", "Unemployed"],
        N,
        p=[0.28, 0.30, 0.20, 0.14, 0.08],
    )
    monthly_income = np.clip(rng.lognormal(np.log(180_000), 0.65, N), 35_000, 4_000_000)
    loan_amount = np.clip(
        monthly_income * rng.uniform(0.25, 1.80, N) + rng.normal(0, 70_000, N),
        20_000,
        2_500_000,
    ).round(-2)
    loan_term = rng.choice([1, 2, 3, 6, 9, 12], N, p=[0.10, 0.08, 0.15, 0.28, 0.17, 0.22])
    previous_loans = np.clip(rng.poisson(2.2, N), 0, 12)
    previous_defaults = np.minimum(previous_loans, rng.binomial(previous_loans, 0.16))
    repayment_history = np.clip(
        94 - previous_defaults * rng.uniform(8, 18, N) + rng.normal(0, 7, N),
        20,
        100,
    )
    savings_balance = np.clip(monthly_income * rng.uniform(0.05, 1.7, N), 0, 3_000_000)
    business_age = np.where(
        np.isin(employment_type, ["Self-employed", "Trader"]),
        rng.integers(1, 181, N),
        rng.integers(0, 97, N),
    )
    daily_sales = np.where(
        np.isin(employment_type, ["Self-employed", "Trader"]),
        np.clip(monthly_income / 26 * rng.uniform(0.55, 1.7, N), 0, 500_000),
        np.clip(rng.normal(0, 1, N), 0, None),
    )
    dti = np.clip(
        loan_amount / np.maximum(monthly_income * np.maximum(loan_term, 1), 1),
        0.02,
        2.0,
    )
    collateral = rng.choice(["Yes", "No"], N, p=[0.30, 0.70])
    region = rng.choice(
        ["South South", "South West", "South East", "North Central", "North West", "North East"],
        N,
        p=[0.18, 0.25, 0.17, 0.16, 0.14, 0.10],
    )

    # Educational synthetic target-generation rule.
    # It deliberately makes repayment history, previous defaults and affordability
    # important, while leaving noise so the model has a meaningful learning task.
    risk_logit = (
        -2.0
        + 1.35 * previous_defaults
        + 1.55 * np.maximum(dti - 0.35, 0)
        - 0.035 * (repayment_history - 70)
        - 0.00000035 * savings_balance
        - 0.0015 * business_age
        + 0.00000035 * np.maximum(loan_amount - monthly_income, 0)
        + 0.35 * (employment_type == "Casual")
        + 0.55 * (employment_type == "Unemployed")
        - 0.25 * (collateral == "Yes")
        + 0.25 * (loan_term >= 12)
        + rng.normal(0, 0.85, N)
    )
    probability = sigmoid(risk_logit)
    defaulted = rng.binomial(1, probability, N)

    df = pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "employment_type": employment_type,
            "monthly_income_ngn": monthly_income.round(2),
            "loan_amount_ngn": loan_amount,
            "loan_term_months": loan_term,
            "previous_loans": previous_loans,
            "previous_defaults": previous_defaults,
            "repayment_history_score": repayment_history.round(2),
            "savings_balance_ngn": savings_balance.round(2),
            "business_age_months": business_age,
            "daily_sales_ngn": daily_sales.round(2),
            "debt_to_income_ratio": dti.round(4),
            "collateral_available": collateral,
            "region": region,
            "defaulted": defaulted,
        }
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Created {OUT} with shape={df.shape}")
    print(df["defaulted"].value_counts(normalize=True).rename("share"))


if __name__ == "__main__":
    main()
