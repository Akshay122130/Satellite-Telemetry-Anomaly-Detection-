from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import numpy as np
import os
import logging

# Import the integrated pipeline
from pipeline import AnomalyDetectionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TelemetryAPI")

# Initialize FastAPI app
app = FastAPI(
    title="Spacecraft Telemetry Anomaly Detection API",
    description="Full novelty integration for real-time telemetry analysis and health scoring.",
    version="1.0.0"
)

# Initialize the pipeline as a global object for reuse
# This avoids reloading models on every request
try:
    pipeline = AnomalyDetectionPipeline(
        rf_path=os.getenv('RF_MODEL_PATH', 'model/rf_model.pkl'),
        xgb_path=os.getenv('XGB_MODEL_PATH', 'model/xgb_model.pkl')
    )
except Exception as e:
    logger.error(f"Failed to initialize pipeline: {e}")
    pipeline = None

# --- Request/Response Schemas ---

class TelemetryRequest(BaseModel):
    temperature: float = Field(..., description="Temperature sensor reading.")
    voltage: float = Field(..., description="Voltage sensor reading.")
    current: float = Field(..., description="Current sensor reading.")
    battery: float = Field(..., description="Battery state-of-charge percentage (0-100).")
    context: str = Field("normal", description="Environmental context: 'sunlight', 'shadow', or 'normal'.")

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        if not (-273.15 <= v <= 1000): # Basic physical sanity check
            raise ValueError("Temperature out of realistic range.")
        return v

    @field_validator('voltage')
    @classmethod
    def validate_voltage(cls, v):
        if v < 0:
            raise ValueError("Voltage cannot be negative.")
        return v

    @field_validator('current')
    @classmethod
    def validate_current(cls, v):
        if not (-100 <= v <= 100): # Example range
            raise ValueError("Current out of realistic range.")
        return v

    @field_validator('battery')
    @classmethod
    def validate_battery(cls, v):
        if not (0 <= v <= 100):
            raise ValueError("Battery must be between 0 and 100.")
        return v

    @field_validator('context')
    @classmethod
    def validate_context(cls, v):
        allowed = ["sunlight", "shadow", "normal"]
        if v.lower() not in allowed:
            raise ValueError(f"Invalid context. Allowed: {allowed}")
        return v.lower()

class PredictionResponse(BaseModel):
    temperature: float
    voltage: float
    current: float
    battery: float
    prediction: str
    probability: float
    threshold: float
    health_score: float
    status: str
    alert: str

# --- Endpoints ---

@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    """
    Health check endpoint.
    """
    return {
        "status": "API Running",
        "system": "Telemetry Anomaly Detection",
        "pipeline_ready": pipeline is not None
    }

@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(request: TelemetryRequest):
    """
    Predicts anomalies for structured telemetry sensors using the integrated pipeline.
    
    Pipeline Flow & Novelty Modules:
    1. Ingestion: Accepts individual sensor readings (Temp, Volt, Curr, Batt) and context.
    2. Preprocessing: Standardizes values using StandardScaler.
    3. Feature Engineering: Expands input into engineered features (rolling stats, etc.).
    4. Contextual Adjustment: Adjusts probabilities based on sunlight/shadow context.
    5. Fusion Model: Weighted ensemble of Random Forest (0.7) and XGBoost (0.3).
    6. Dynamic Thresholding: Uses adaptive percentiles for final classification.
    7. Health Scoring: Converts risk into a 0-100 system health index.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Anomaly detection pipeline is not initialized. Check server logs."
        )

    try:
        # Run the pipeline with structured inputs
        results = pipeline.run(
            temperature=request.temperature,
            voltage=request.voltage,
            current=request.current,
            battery=request.battery,
            context=request.context
        )
        
        prediction = results['prediction']
        health_score = float(results['health_score'])
        
        # 8. Determine system status based on health score
        if health_score > 80:
            system_status = "Healthy"
        elif health_score > 50:
            system_status = "Warning"
        else:
            system_status = "Critical"
            
        # 9. Determine alert message based on prediction
        alert_msg = "Anomaly Detected" if prediction == "Anomaly" else "System Stable"
        
        # Return structured JSON response including original sensor values
        return {
            "temperature": request.temperature,
            "voltage": request.voltage,
            "current": request.current,
            "battery": request.battery,
            "prediction": prediction,
            "probability": float(results['probability']),
            "threshold": float(results['threshold']),
            "health_score": health_score,
            "status": system_status,
            "alert": alert_msg
        }

    except ValueError as ve:
        logger.error(f"Input Validation Error: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Pipeline Execution Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during telemetry processing."
        )

if __name__ == "__main__":
    import uvicorn
    # Using port 8003 to avoid address-in-use errors
    uvicorn.run(app, host="0.0.0.0", port=8003)
