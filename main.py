import os
from datetime import datetime
from pathlib import Path
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Bert Based Model
# model_path = os.path.join(BASE_DIR, "bert_spam_model")
# tokenizer = BertTokenizer.from_pretrained(model_path)
# model = BertForSequenceClassification.from_pretrained(model_path)
# model.eval()

model_name = "Anurag3703/bert-spam-classifier"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# FastAPI app
app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve React frontend
frontend_build_path = Path(__file__).parent.parent / "frontend" / "build"
print()
print(f"Frontend path: {frontend_build_path}")
print()

if frontend_build_path.exists():
    app.mount("/static", StaticFiles(directory=frontend_build_path / "static"), name="static")


@app.get("/")
async def serve_frontend():
    """Serves the React app's index.html file."""
    index_path = frontend_build_path / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"error": "Frontend build not found. Run npm run build in the frontend folder."}


# API Root
@app.get("/api")
def read_root():
    return {"message": "Welcome to the Scam/Ham Prediction API"}


# Predict Endpoint
class Message(BaseModel):
    message: str


@app.post("/predict")
async def predict(message: Message):
    try:
        inputs = tokenizer(
            message.message,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            output = model(**inputs)
        prediction_num = torch.argmax(output.logits).item()

        prediction_text = "Spam" if prediction_num == 1 else "Ham"

        # Log results without storing in DB
        result = {
            "message": message.message,
            "prediction": prediction_text,
            "prediction_num": prediction_num,
            "timestamp": datetime.now().isoformat()
        }
        print(f"Prediction result: {result}")

        # Return both numeric and text prediction
        return {"prediction": prediction_text, "prediction_num": prediction_num}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run FastAPI (for local development)
if __name__ == "__main__":
    import uvicorn

    print("Starting server with BERT model only (no database connections)")
    print("API will be available at http://127.0.0.1:8001")
    # Fix the warning by not using reload=True
    uvicorn.run("main:app", host="127.0.0.1", port=8001)
