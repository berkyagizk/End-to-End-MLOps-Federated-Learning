from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import joblib 
import pandas as pd

app = FastAPI(title="Credit Card Fraud Detection API", version="1.0")

# MODEL YÜKLEME KISMI
try:
    model = joblib.load(Path(__file__).resolve().parent / "best_model.pkl")
    print("Model best_model.pkl dosyasından başarıyla yüklendi!")
except Exception as e:
    print(f"Model yüklenirken hata oluştu: {e}")
    model = None

#  VERİ ŞEMALARI 
class TransactionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    is_fraud: bool
    prediction: int

#  ENDPOINTLER 
@app.get("/")
def read_root():
    return {"message": "Federe Öğrenme Fraud Detection Servisi Ayakta."}

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(request: TransactionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model bulunamadı.")
    
    try:
        input_data = pd.DataFrame([request.features])
        prediction = model.predict(input_data)
        result = int(prediction[0])
        is_fraud = bool(result == 1)
        
        return PredictionResponse(is_fraud=is_fraud, prediction=result)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))