"""
FastAPI app for serving EfficientNetB0 EuroSAT model predictions.
Allows users to upload an RGB image and returns the predicted land use class and confidence score.
"""
import os
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/efficientnetb0_best.pt')
CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]
IMG_SIZE = (64, 64)

app = FastAPI(title="EuroSAT Land Use Classification API")

# Model and transform setup
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

model = load_model()
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img)
            probs = F.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            class_name = CLASS_NAMES[pred.item()]
            confidence = conf.item()
        return JSONResponse({
            "predicted_class": class_name,
            "confidence": round(confidence, 4)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/")
def root():
    return {"message": "EuroSAT Land Use Classification API. Use /predict to POST an image."}
