"""
FastAPI app for serving EfficientNetB0 EuroSAT model predictions.
Allows users to upload an RGB image and returns the predicted land use class and confidence score.
"""
import os
import io
import sys
import base64
from datetime import datetime, timedelta
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
import rasterio
from rasterio.windows import Window

# Add scripts to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts'))

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/efficientnetb0_best.pt')
CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]
IMG_SIZE = (64, 64)

app = FastAPI(title="EuroSAT Land Use Classification API")

# Enable CORS for HTML frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            
            # Get all class probabilities
            all_probs = {CLASS_NAMES[i]: round(probs[0][i].item(), 4) for i in range(len(CLASS_NAMES))}
            
        return JSONResponse({
            "predicted_class": class_name,
            "confidence": round(confidence, 4),
            "all_probabilities": all_probs
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

class SearchRequest(BaseModel):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    start_date: str
    end_date: str
    max_cloud: int = 10

class LoadImageRequest(BaseModel):
    item_id: str
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    start_date: str
    end_date: str
    max_cloud: int = 20

@app.post("/search-sentinel")
async def search_sentinel(request: SearchRequest):
    """Search for Sentinel-2 images in the specified bounding box."""
    try:
        from sentinel_search import search_sentinel2_images, get_item_metadata  # type: ignore
        
        bbox = [request.min_lon, request.min_lat, request.max_lon, request.max_lat]
        results = search_sentinel2_images(
            bbox,
            request.start_date,
            request.end_date,
            request.max_cloud
        )
        
        # Return metadata for each result
        items = []
        for item in results[:10]:  # Limit to 10 results
            metadata = get_item_metadata(item)
            items.append({
                "id": metadata["id"],
                "datetime": metadata["datetime"],
                "cloud_cover": metadata["cloud_cover"]
            })
        
        return JSONResponse({"items": items})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/load-sentinel")
async def load_sentinel(request: LoadImageRequest):
    """Load a specific Sentinel-2 image and return as base64 with preprocessing."""
    try:
        from sentinel_search import search_sentinel2_images, get_rgb_image_url  # type: ignore
        
        # Re-search using the user's original parameters
        bbox = [request.min_lon, request.min_lat, request.max_lon, request.max_lat]
        results = search_sentinel2_images(
            bbox,
            request.start_date,
            request.end_date,
            request.max_cloud
        )
        
        # Find the matching item
        item = None
        for result in results:
            if result.id == request.item_id:
                item = result
                break
        
        if not item:
            return JSONResponse({"error": "Item not found"}, status_code=404)
        
        url_data = get_rgb_image_url(item)
        
        # Load bands and find valid region
        with rasterio.open(url_data["B04"]) as src:
            height, width = src.height, src.width
            
            # Smart region detection
            search_positions = []
            for row_frac in [0.2, 0.35, 0.5, 0.65, 0.8]:
                for col_frac in [0.2, 0.35, 0.5, 0.65, 0.8]:
                    search_positions.append((int(height * row_frac), int(width * col_frac)))
            
            best_position = None
            best_mean = 0
            
            for row_center, col_center in search_positions:
                row_off = max(0, min(height - 900, row_center - 450))
                col_off = max(0, min(width - 1600, col_center - 800))
                window_height = min(900, height - row_off)
                window_width = min(1600, width - col_off)
                
                test_window = Window(col_off + 256, row_off + 256, min(64, window_width - 256), min(64, window_height - 256))
                test_data = src.read(1, window=test_window)
                mean_val = test_data.mean()
                
                if mean_val > best_mean:
                    best_mean = mean_val
                    best_position = (row_off, col_off, window_height, window_width)
                
                if mean_val > 100:
                    break
            
            row_off, col_off, window_height, window_width = best_position
        
        # Load all bands
        bands = []
        for band_name in ["B04", "B03", "B02"]:
            with rasterio.open(url_data[band_name]) as src:
                window = Window(col_off, row_off, window_width, window_height)
                band = src.read(1, window=window)
                bands.append(band)
        
        rgb = np.stack(bands, axis=-1)
        
        # Create display version (0-255)
        rgb_display = np.clip(rgb, 0, 2750)
        rgb_display = ((rgb_display / 2750.0) * 255).astype(np.uint8)
        
        # Convert to base64
        img_display = Image.fromarray(rgb_display)
        buffer = io.BytesIO()
        img_display.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return JSONResponse({
            "image": img_base64,
            "width": window_width,
            "height": window_height
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/predict-region")
async def predict_region(data: dict):
    """Predict land use for a specific region of a Sentinel-2 image."""
    try:
        # Extract base64 image and crop coordinates
        img_base64 = data["image"]
        x = data["x"]
        y = data["y"]
        
        # Decode image (this is already display version 0-255)
        img_bytes = base64.b64decode(img_base64)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Extract 64x64 crop FIRST
        crop = img.crop((x, y, x + 64, y + 64))
        
        # Convert crop to numpy for EuroSAT preprocessing
        rgb = np.array(crop)
        
        # The image coming from /load-sentinel is display version (0-255)
        # We need to reverse it back to raw values, then apply model preprocessing
        # Display: rgb_display = (raw / 2750) * 255
        # Reverse: raw = (rgb_display / 255) * 2750
        # Then apply model: rgb_model = (raw / 2750) * 254 + 1
        # Simplified: rgb_model = (rgb_display / 255) * 254 + 1
        
        rgb_model = ((rgb.astype(np.float32) / 255.0) * 254 + 1).astype(np.uint8)
        crop_model = Image.fromarray(rgb_model)
        
        # Apply transforms and predict
        crop_tensor = transform(crop_model).unsqueeze(0)
        with torch.no_grad():
            outputs = model(crop_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            class_name = CLASS_NAMES[pred.item()]
            confidence = conf.item()
            
            # Get all class probabilities
            all_probs = {CLASS_NAMES[i]: round(probs[0][i].item(), 4) for i in range(len(CLASS_NAMES))}
        
        return JSONResponse({
            "predicted_class": class_name,
            "confidence": round(confidence, 4),
            "all_probabilities": all_probs
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/")
def root():
    return {"message": "EuroSAT Land Use Classification API. Use /predict to POST an image."}
