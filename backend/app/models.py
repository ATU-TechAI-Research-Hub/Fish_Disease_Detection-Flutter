from pydantic import BaseModel


class Disease(BaseModel):
    id: int
    name: str
    type: str
    cause: str
    symptoms: str
    treatment: str
    prevention: str


class ClassProbability(BaseModel):
    disease_id: int
    disease_name: str
    confidence: float


class PredictionResponse(BaseModel):
    prediction: Disease
    confidence: float
    source: str
    filename: str
    top_predictions: list[ClassProbability] = []
