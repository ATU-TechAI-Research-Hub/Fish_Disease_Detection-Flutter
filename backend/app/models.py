from pydantic import BaseModel


class Disease(BaseModel):
    id: int
    name: str
    type: str
    cause: str
    symptoms: str
    treatment: str
    prevention: str


class PredictionResponse(BaseModel):
    prediction: Disease
    confidence: float
    source: str
    filename: str
