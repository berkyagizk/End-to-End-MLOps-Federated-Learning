from contextlib import asynccontextmanager
from io import BytesIO
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, StrictFloat
import joblib
import numpy as np
import pandas as pd

from src.preprocessing import FEATURE_NAMES, load_model_preprocessor


def load_inference_bundle(model_path):
    # Only use model files explicitly configured by a trusted operator.
    model_bytes = Path(model_path).read_bytes()
    preprocessor = load_model_preprocessor(model_path, model_bytes)
    model = joblib.load(BytesIO(model_bytes))
    if not callable(getattr(model, "predict", None)) or getattr(model, "n_features_in_", None) != len(FEATURE_NAMES):
        raise ValueError(f"Expected a fitted prediction model with {len(FEATURE_NAMES)} features")
    if hasattr(model, "feature_names_in_") and tuple(model.feature_names_in_) != FEATURE_NAMES:
        raise ValueError("Model feature names or order do not match the preprocessing schema")
    return model, preprocessor


@asynccontextmanager
async def lifespan(app):
    app.state.bundle = None
    model_path = os.getenv("FRAUD_MODEL_PATH")
    if model_path:
        try:
            app.state.bundle = load_inference_bundle(model_path)
        except Exception:
            logging.getLogger(__name__).exception("Could not load the configured model and preprocessing metadata")
    try:
        yield
    finally:
        app.state.bundle = None


app = FastAPI(title="Credit Card Fraud Detection API", version="1.0", lifespan=lifespan)


class TransactionRequest(BaseModel):
    features: list[StrictFloat]


class PredictionResponse(BaseModel):
    is_fraud: bool
    prediction: int


@app.get("/")
def read_root():
    return {"message": "Federe Öğrenme Fraud Detection Servisi Ayakta."}

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: TransactionRequest, request: Request):
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="A model with compatible preprocessing metadata is unavailable.")

    model, preprocessor = bundle
    try:
        input_data = preprocessor.transform(np.asarray([transaction.features]))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    if hasattr(model, "feature_names_in_"):
        input_data = pd.DataFrame(input_data, columns=FEATURE_NAMES)
    result = int(model.predict(input_data)[0])
    return PredictionResponse(is_fraud=result == 1, prediction=result)
