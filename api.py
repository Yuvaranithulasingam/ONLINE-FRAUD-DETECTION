from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load the trained model
import joblib

# Load the trained fraud detection model
model = joblib.load(r"C:\Users\thula\OneDrive\Documents\GIDY\fraud_detection_model.pkl")

@app.post("/predict/")
async def predict(features: list):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return {"fraud": int(prediction[0])}

# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
