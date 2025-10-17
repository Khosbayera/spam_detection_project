from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI(title="Unified Spam Detection API", version="1.0")

# Load models
call_model = joblib.load("models/call_spam_model.pkl")
sms_model = joblib.load("models/sms_spam_model.pkl")
sms_vectorizer = joblib.load("models/sms_vectorizer.pkl")

# Request models
class CallData(BaseModel):
    calls_last_24h: int
    calls_last_7d: int
    avg_duration_seconds: int
    ratio_missed_calls: float
    is_in_contacts: int

class SMSData(BaseModel):
    message: str

@app.get("/")
def home():
    return {"message": "Unified Spam Detection API is running!"}

@app.post("/predict_call/")
def predict_call(data: CallData):
    features = np.array([[data.calls_last_24h, data.calls_last_7d, data.avg_duration_seconds,
                          data.ratio_missed_calls, data.is_in_contacts]])
    prob = call_model.predict_proba(features)[0]
    prediction = int(prob[1] > 0.5)
    return {
        "type": "call",
        "prediction": "Spam" if prediction == 1 else "Not Spam",
        "confidence": round(float(prob[prediction]), 2)
    }

@app.post("/predict_sms/")
def predict_sms(data: SMSData):
    features = sms_vectorizer.transform([data.message])
    prob = sms_model.predict_proba(features)[0]
    prediction = int(prob[1] > 0.5)
    return {
        "type": "sms",
        "prediction": "Spam" if prediction == 1 else "Not Spam",
        "confidence": round(float(prob[prediction]), 2)
    }

@app.post("/predict/")
def predict_unified(call_data: CallData = None, sms_data: SMSData = None):
    results = {}
    if call_data:
        call_features = np.array([[call_data.calls_last_24h, call_data.calls_last_7d, 
                                   call_data.avg_duration_seconds, call_data.ratio_missed_calls, 
                                   call_data.is_in_contacts]])
        prob = call_model.predict_proba(call_features)[0]
        prediction = int(prob[1] > 0.5)
        results['call'] = {"prediction": "Spam" if prediction == 1 else "Not Spam",
                           "confidence": round(float(prob[prediction]), 2)}
    if sms_data:
        sms_features = sms_vectorizer.transform([sms_data.message])
        prob = sms_model.predict_proba(sms_features)[0]
        prediction = int(prob[1] > 0.5)
        results['sms'] = {"prediction": "Spam" if prediction == 1 else "Not Spam",
                          "confidence": round(float(prob[prediction]), 2)}
    return results
