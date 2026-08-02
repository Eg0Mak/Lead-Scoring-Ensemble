from typing import Any

from pydantic import BaseModel, ConfigDict


class LeadRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    lead_id: str
    assignment_ts: str


class PredictRequest(BaseModel):
    records: list[LeadRecord]


class PredictionItem(BaseModel):
    lead_id: str
    conversion_probability: float


class PredictResponse(BaseModel):
    predictions: list[PredictionItem]