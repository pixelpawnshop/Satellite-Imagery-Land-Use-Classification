# EuroSAT Land Use Classifier - Web Application

🛰️ **Live Demo:** [https://pixelpawnshop.github.io/Satellite-Imagery-Land-Use-Classification/](https://pixelpawnshop.github.io/Satellite-Imagery-Land-Use-Classification/)

## Overview

This is an interactive web application for classifying land use from satellite imagery using deep learning. The app allows users to search for Sentinel-2 satellite images, select regions of interest, and get real-time land use predictions powered by an EfficientNetB0 model trained on the EuroSAT dataset.

https://github.com/user-attachments/assets/9f2b9ec1-747a-40e7-bdcb-95d1950d11db

---

## Architecture

### Frontend (GitHub Pages)
- **Hosting:** Static HTML/CSS/JavaScript hosted on GitHub Pages
- **Framework:** Vanilla JavaScript with modern ES6+ features
- **Features:**
  - Interactive search interface for Sentinel-2 imagery via Microsoft Planetary Computer API
  - Canvas-based image viewer with draggable 64×64 region selector
  - Real-time predictions with confidence scores and probability distributions
  - Dark/Light theme toggle
  - Responsive design with collapsible sidebar

### Backend (Render - Free Tier)
- **Hosting:** Docker container deployed on Render's free tier
- **Framework:** FastAPI (Python)
- **Model:** EfficientNetB0 fine-tuned on EuroSAT dataset
- **Endpoints:**
  - `/search-sentinel` - Search for Sentinel-2 images by location, date, and cloud cover
  - `/load-sentinel` - Load and preprocess a specific Sentinel-2 image
  - `/predict-region` - Classify a 64×64 region from a loaded image

---

## How It Works

### 1. Image Search
Users enter geographic coordinates (longitude/latitude), date range, and maximum cloud cover percentage. The frontend sends a request to the backend, which queries the Microsoft Planetary Computer STAC API for matching Sentinel-2 L2A imagery.

### 2. Image Loading
When a user selects an image from the search results, the backend:
- Fetches the raw Sentinel-2 bands (B04, B03, B02) from the cloud
- Applies EuroSAT preprocessing (clipping to 0-2750, scaling to 1-255)
- Finds a valid data region (non-black/nodata)
- Returns a 1024×1024 PNG image to the frontend

### 3. Region Selection
The frontend displays the image on an HTML5 canvas with a draggable red 64×64 box. Users can:
- Click to reposition the box
- Drag the box to their area of interest
- View the exact pixel coordinates in real-time

### 4. Prediction
When the user clicks "Predict Land Use":
- The frontend extracts the 64×64 region coordinates
- Sends the full image and coordinates to the backend
- Backend crops the region, preprocesses it, and runs inference with the trained model
- Returns the predicted class, confidence score, and all class probabilities
- Frontend displays results with an interactive bar chart

---

## Technology Stack

### Frontend
- **HTML5 Canvas** for image rendering and interaction
- **CSS Variables** for theming (light/dark mode)
- **Fetch API** for asynchronous backend communication
- **LocalStorage** for theme persistence

### Backend
- **FastAPI** - Modern, fast web framework for APIs
- **PyTorch** - Deep learning framework for model inference
- **Rasterio** - Geospatial raster I/O for Sentinel-2 data
- **Planetary Computer SDK** - Access to Sentinel-2 imagery
- **Docker** - Containerized deployment

---

## Model Details

- **Architecture:** EfficientNetB0 (transfer learning from ImageNet)
- **Dataset:** EuroSAT (27,000 labeled Sentinel-2 image patches)
- **Classes:** 10 land use categories (AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake)
- **Test Accuracy:** 80.74%
- **Input Size:** 64×64 RGB images
- **Preprocessing:** EuroSAT standard (bands B04/B03/B02, scaled 0-2750 → 1-255)

---

## Deployment

### Frontend (GitHub Pages)
This branch (`gh-pages`) contains only the static frontend files:
- `index.html` - Complete single-page application
- Automatically deployed via GitHub Pages
- No build step required

### Backend (Render)
The main branch contains the full project including:
- FastAPI server (`serving/app.py`)
- Trained model (`models/efficientnetb0_best.pt`)
- Docker configuration (`serving/Dockerfile`, `docker-compose.yml`)
- Deployed on Render's free tier with automatic Docker builds

**Note:** The free tier spins down after 15 minutes of inactivity. First requests may take 30-60 seconds while the server wakes up.

---

## API Reference

### Search Sentinel-2 Images
```
POST https://satellite-imagery-land-use-classification.onrender.com/search-sentinel
Content-Type: application/json

{
  "min_lon": 13.0,
  "min_lat": 52.0,
  "max_lon": 14.0,
  "max_lat": 53.0,
  "start_date": "2023-06-01",
  "end_date": "2023-06-30",
  "max_cloud": 10
}
```

### Load Sentinel-2 Image
```
POST https://satellite-imagery-land-use-classification.onrender.com/load-sentinel
Content-Type: application/json

{
  "item_id": "S2A_...",
  "min_lon": 13.0,
  "min_lat": 52.0,
  "max_lon": 14.0,
  "max_lat": 53.0,
  "start_date": "2023-06-01",
  "end_date": "2023-06-30",
  "max_cloud": 10
}
```

### Predict Land Use
```
POST https://satellite-imagery-land-use-classification.onrender.com/predict-region
Content-Type: application/json

{
  "image": "base64_encoded_image",
  "x": 480,
  "y": 480
}
```

---

## Features

✅ Real-time Sentinel-2 image search  
✅ Interactive canvas-based region selection  
✅ Deep learning-powered land use classification  
✅ Confidence scores and probability distributions  
✅ Dark/Light theme support  
✅ Responsive design  
✅ No installation required - runs entirely in the browser  

---

## Performance

- **Frontend:** Instant loading (static HTML)
- **Image Search:** ~2-5 seconds (depends on Microsoft Planetary Computer API)
- **Image Loading:** ~5-10 seconds (downloads and processes Sentinel-2 bands)
- **Prediction:** <1 second (model inference)
- **First Request:** Up to 60 seconds (Render free tier cold start)

---

## Browser Compatibility

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Opera 76+

Requires JavaScript enabled and support for:
- HTML5 Canvas
- CSS Variables
- Fetch API
- ES6+ (async/await, arrow functions, template literals)

---

## Credits

- **Developer:** pixelpawnshop
- **Dataset:** [EuroSAT](https://github.com/phelber/eurosat) by Helber et al.
- **Satellite Imagery:** [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/)
- **Model:** EfficientNetB0 by Tan & Le

---

## Repository

**Main Project:** [Satellite-Imagery-Land-Use-Classification (main branch)](https://github.com/pixelpawnshop/Satellite-Imagery-Land-Use-Classification)  
**Frontend Source:** This branch (`gh-pages`)  
**Backend Deployment:** [Render](https://satellite-imagery-land-use-classification.onrender.com/)

---

## License

MIT License
