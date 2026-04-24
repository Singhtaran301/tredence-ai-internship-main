import io
import os
from pathlib import Path

import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from solution import CIFAR10_CLASSES, SelfPruningNet


MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts_final/models/model_lambda_5.pth"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Self-Pruning CIFAR-10 Classifier")

model = SelfPruningNet().to(DEVICE)
if MODEL_PATH.exists():
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
else:
    model = None

transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
)


@app.get("/")
async def healthcheck() -> dict:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model file not found at {MODEL_PATH}")

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        predicted_index = logits.argmax(dim=1).item()

    return {
        "class_index": predicted_index,
        "class_name": CIFAR10_CLASSES[predicted_index],
    }
