from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model" / "loan_default_pipeline.joblib"
DATA = ROOT / "data" / "loan_default_nigeria_synthetic.csv"


def test_dataset_exists_and_has_target():
    assert DATA.exists()
    df = pd.read_csv(DATA)
    assert len(df) >= 1000
    assert "defaulted" in df.columns
    assert set(df["defaulted"].unique()).issubset({0, 1})


def test_model_loads_and_predicts():
    model = joblib.load(MODEL)
    df = pd.read_csv(DATA).drop(columns=["defaulted"]).head(5)
    probabilities = model.predict_proba(df)[:, 1]
    assert len(probabilities) == 5
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
