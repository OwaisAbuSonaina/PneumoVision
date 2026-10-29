# app.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
import uvicorn
from huggingface_hub import hf_hub_download
import os
from pathlib import Path
from model_service import PneumoniaDetector

MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models"))
MODEL_FILENAME = "best_model.keras"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME

inference_engine = None

@asynccontextmanager
async def lifespan(app:FastAPI):
    global inference_engine

    try: 
        inference_engine = PneumoniaDetector(model_path=MODEL_PATH)
        print("MODEL LOADED SUCCESSFULLY.")
    except Exception as e: 
        print(f"ERROR LOADING MODEL: {e}")

    yield


app = FastAPI(title="PneumoVision API", lifespan=lifespan)


@app.get("/")
def home():
    return {"message": "PneumoVision API is running."}

@app.post("/predict")
async def predict(file:UploadFile=File(...)):

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="File must be JPG or PNG")

    try:
        image_bytes = await file.read()
        prediction = inference_engine.predict(image=image_bytes)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, port=8000)