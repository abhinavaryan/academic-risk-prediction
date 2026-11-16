from fastapi import FastAPI, Request, Form
from fastapi.templating import JinjaTemplates
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI()
templates = JinjaTemplates(directory="templates")

# Load real trained model and features from the model folder
model_path = os.path.join(os.path.dirname(__file__), '../model/risk_model.pkl')
features_path = os.path.join(os.path.dirname(__file__), '../model/model_features.pkl')

model = joblib.load(model_path)
model_features = joblib.load(features_path)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "prediction": None
    })

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    gender: str = Form(...),
    region: str = Form(...),
    highest_education: str = Form(...),
    imd_band: str = Form(...),
    age_band: str = Form(...),
    num_of_prev_attempts: int = Form(...),
    studied_credits: int = Form(...),
    disability: str = Form(...)
):
    # Construct input dataframe matching training columns dynamically
    input_data = pd.DataFrame(columns=model_features)
    input_data.loc[0] = 0 # Initialize with zeros
    
    # Safely assign values if columns exist in training features
    try:
        if f"gender_{gender}" in input_data.columns:
            input_data.loc[0, f"gender_{gender}"] = 1
        if f"region_{region}" in input_data.columns:
            input_data.loc[0, f"region_{region}"] = 1
        if f"highest_education_{highest_education}" in input_data.columns:
            input_data.loc[0, f"highest_education_{highest_education}"] = 1
        if f"imd_band_{imd_band}" in input_data.columns:
            input_data.loc[0, f"imd_band_{imd_band}"] = 1
        if f"age_band_{age_band}" in input_data.columns:
            input_data.loc[0, f"age_band_{age_band}"] = 1
        if "num_of_prev_attempts" in input_data.columns:
            input_data.loc[0, "num_of_prev_attempts"] = num_of_prev_attempts
        if "studied_credits" in input_data.columns:
            input_data.loc[0, "studied_credits"] = studied_credits
        if f"disability_{disability}" in input_data.columns:
            input_data.loc[0, f"disability_{disability}"] = 1
    except Exception as e:
        pass

    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][prediction])
    
    result_text = "Low Risk (Student is safe & likely to Pass)" if prediction == 1 else "High Risk (Student requires immediate Academic Intervention)"
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "prediction": result_text, 
        "confidence": f"{probability * 100:.2f}%",
        "is_safe": prediction == 1
    })