# tests/test_api_sample.py
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from src.main import app

TEST_CSV = PROJECT_ROOT / "data" / "test.csv"
N_SAMPLE = 5
RANDOM_SEED = 42


def test_predict_on_random_test_sample():
    assert TEST_CSV.exists(), f"Не найден файл {TEST_CSV}"

    df = pd.read_csv(TEST_CSV)
    assert len(df) > 0, "test.csv пустой"

    sample = df.sample(n=min(N_SAMPLE, len(df)), random_state=RANDOM_SEED)

    # NaN → null, без ValueError при сериализации JSON
    records = pd.DataFrame(sample).astype(object)
    records = records.where(pd.notnull(records), None).to_dict(orient="records")

    with TestClient(app) as client:
        resp = client.post("/predict", json={"records": records})

    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert "predictions" in body
    preds = body["predictions"]
    assert len(preds) == len(records)

    sample_ids = set(sample["lead_id"].astype(str))
    for p in preds:
        assert "lead_id" in p
        assert "conversion_probability" in p
        assert str(p["lead_id"]) in sample_ids
        score = float(p["conversion_probability"])
        assert 0.0 <= score <= 1.0, f"score вне [0, 1]: {score}"