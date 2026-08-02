import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Request

from .predictor import predict
from .preprocessing import add_business_features, build_event_features_advanced
from .schemas import PredictRequest, PredictResponse

EVENTS_PATH = os.environ.get("EVENTS_PATH", "data/events.csv")


def load_events() -> pd.DataFrame:
    return pd.read_csv(EVENTS_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.df_events = load_events()
    yield


app = FastAPI(
    title="Lead Scoring API",
    version="1.0.0",
    description="API for lead conversion probability prediction",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"service": "Lead Scoring API", "status": "running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest, req: Request):
    df = pd.DataFrame([r.model_dump() for r in request.records])

    event_feat = build_event_features_advanced(df, req.app.state.df_events, extra_v2=True)
    df = df.merge(event_feat, on="lead_id", how="left")
    df = add_business_features(df)

    scores = predict(df)

    return {
        "predictions": [
            {"lead_id": record.lead_id, "conversion_probability": float(score)}
            for record, score in zip(request.records, scores)
        ]
    }