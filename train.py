from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "loan_default_nigeria_synthetic.csv"
MODEL_DIR = ROOT / "model"

NUMERIC = [
    "age", "monthly_income_ngn", "loan_amount_ngn", "loan_term_months",
    "previous_loans", "previous_defaults", "repayment_history_score",
    "savings_balance_ngn", "business_age_months", "daily_sales_ngn",
    "debt_to_income_ratio",
]
CATEGORICAL = ["gender", "employment_type", "collateral_available", "region"]


def build_preprocessor():
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, NUMERIC),
            ("cat", categorical_pipe, CATEGORICAL),
        ]
    )


def make_pipeline(model):
    return Pipeline([("preprocessor", build_preprocessor()), ("model", model)])


def choose_threshold(y_true, probabilities):
    thresholds = np.linspace(0.10, 0.90, 161)
    scores = [f1_score(y_true, probabilities >= t, zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def main():
    df = pd.read_csv(DATA)
    X = df[NUMERIC + CATEGORICAL]
    y = df["defaulted"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    candidates = {
        "logistic_regression": make_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        ),
        "random_forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=350,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        ),
    }

    results = {}
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        p = pipe.predict_proba(X_val)[:, 1]
        results[name] = {
            "roc_auc": float(roc_auc_score(y_val, p)),
            "threshold": choose_threshold(y_val, p),
        }

    selected_name = max(results, key=lambda k: results[k]["roc_auc"])
    selected = candidates[selected_name]

    # Refit on train+validation, then lock the model before final test.
    X_fit = pd.concat([X_train, X_val])
    y_fit = pd.concat([y_train, y_val])
    selected.fit(X_fit, y_fit)

    test_prob = selected.predict_proba(X_test)[:, 1]
    threshold = results[selected_name]["threshold"]
    test_pred = (test_prob >= threshold).astype(int)

    metrics = {
        "selected_model": selected_name,
        "validation_results": results,
        "test_roc_auc": float(roc_auc_score(y_test, test_prob)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "decision_threshold": float(threshold),
        "test_classification_report": classification_report(
            y_test, test_pred, output_dict=True, zero_division=0
        ),
        "data_rows": int(len(df)),
        "default_rate": float(y.mean()),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(selected, MODEL_DIR / "loan_default_pipeline.joblib")

    metadata = {
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "all_features": NUMERIC + CATEGORICAL,
        "risk_bands": {
            "low": "< 30",
            "medium": "30-60",
            "high": "> 60",
        },
        "selected_model": selected_name,
        "decision_threshold": threshold,
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    cm = confusion_matrix(y_test, test_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)
    ax.set_title("AI-04 Test Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], ["No Default", "Default"])
    ax.set_yticks([0, 1], ["No Default", "Default"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, test_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC={metrics['test_roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("AI-04 ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "roc_curve.png", dpi=160)
    plt.close(fig)

    # Global feature importance.
    pre = selected.named_steps["preprocessor"]
    model = selected.named_steps["model"]
    names = pre.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    else:
        vals = np.abs(model.coef_[0])

    fi = pd.DataFrame({"feature": names, "importance": vals}).sort_values(
        "importance", ascending=False
    )
    fi.to_csv(MODEL_DIR / "feature_importance.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print("\nTop features:")
    print(fi.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
