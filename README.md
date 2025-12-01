# Satellite Imagery Land Use Classification (EuroSAT)

## Project Structure


## Tech Stack

## Getting Started
1. Clone the repo
2. Install dependencies
3. Download EuroSAT dataset
4. Run preprocessing and training scripts
5. Serve model via FastAPI
6. Visualize results


This project demonstrates MLOps best practices for satellite imagery land use classification using transfer learning and PyTorch.
# EuroSAT Land Use Classification

## Overview
This is an interactive web application for classifying land use from satellite imagery using deep learning. The app allows users to search for Sentinel-2 satellite images, select regions of interest, and get real-time land use predictions powered by an EfficientNetB0 model trained on the EuroSAT dataset.

**Goal:** Deploy a public-facing service so users can access and test land use predictions on satellite images.

https://github.com/user-attachments/assets/f24cf186-e32e-44d1-9c94-a0917dbac0f0

---

## Features
- Download and preprocess EuroSAT dataset
- Train EfficientNetB0 with transfer learning
- Evaluate model with detailed metrics and confusion matrix
- Serve predictions via FastAPI API
- Interactive web apps for image upload and Sentinel-2 search
- Ready for deployment on your own server

---

## Dataset
- **Source:** [EuroSAT](https://github.com/phelber/eurosat)
- **Classes:** AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake
- **Preprocessing:** Images resized to 64x64, normalized, split into train/val/test
- **Scripts:**
	- `scripts/download_eurosat.py`: Download and extract dataset
	- `scripts/preprocess_eurosat.py`: Preprocess and split images

---

## Model
- **Architecture:** EfficientNetB0 (PyTorch, transfer learning)
- **Training:**
	- 10 epochs, Adam optimizer
	- Only classifier head is fine-tuned
	- Data splits: 70% train, 15% val, 15% test
- **Script:** `scripts/train_eurosat.py`

---

## Evaluation
- **Script:** `scripts/evaluate_model.py`
- **Test Accuracy:** **80.74%**
- **Per-class accuracy:** Most classes above 75%, several above 85%
- **Weakest classes:** Highway, River, PermanentCrop (~69-71%)
- **Confusion matrix:** See `reports/confusion_matrix.png`
- **Analysis:**
	- Model is robust for most classes
	- Consider data augmentation or more training for further improvement

---

## API & Web Apps
- **FastAPI backend:** `serving/app.py`
	- `/predict`: Upload RGB image (64x64) for prediction
	- `/search-sentinel`: Search Sentinel-2 images by location/date/cloud cover
	- `/load-sentinel`: Load and preprocess Sentinel-2 image region
- **Streamlit app:** `visualization/streamlit_app.py`
	- Upload images and view predictions
- **HTML/JS frontend:** `visualization/index.html`
	- Interactive Sentinel-2 search and prediction

---

## Deployment
### 1. Install Requirements
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Download & Preprocess Data
```powershell
python scripts/download_eurosat.py
python scripts/preprocess_eurosat.py
```

### 3. Train Model
```powershell
python scripts/train_eurosat.py
```

### 4. Evaluate Model
```powershell
python scripts/evaluate_model.py
```

### 5. Start FastAPI Server
```powershell
uvicorn serving.app:app --host 0.0.0.0 --port 8000
```

### 6. Run Web Apps
- **Streamlit:**
	```powershell
	streamlit run visualization/streamlit_app.py
	```
- **HTML/JS:** Open `visualization/index.html` in your browser

---

## Deployment Best Practices

- **Model file:** The trained model (`models/efficientnetb0_best.pt`) is included in the repository so you do not need to retrain or run any training scripts on the server. FastAPI will load and serve predictions immediately after deployment.
- **Data folder:** The full `data/` folder is not required for serving predictions and is not committed to the repository. This keeps the project lightweight and avoids git performance issues.
- **No retraining required:** You can deploy and run the API and frontend without any extra steps. All necessary files for inference are included.

---

---

## Docker Deployment

### Build and Run with Docker
1. Build the Docker image:
	```powershell
	docker build -t eurosat-api ./serving
	```
2. Run the container:
	```powershell
	docker run -p 8000:8000 eurosat-api
	```

### Using Docker Compose
1. Build and start all services:
	```powershell
	docker-compose up --build
	```
2. Access the API at [http://localhost:8000](http://localhost:8000)

---

---

## Usage
- Upload a 64x64 RGB satellite image to `/predict` endpoint or via web apps
- Search for Sentinel-2 images, select a region, and get predictions
- View per-class probabilities and confidence scores

---

## Folder Structure
```
├── configs/
├── data/
├── experiments/
├── models/
├── notebooks/
├── reports/
├── scripts/
├── serving/
├── visualization/
```

---

## References
- [EuroSAT Dataset](https://github.com/phelber/eurosat)
- [EfficientNet](https://arxiv.org/abs/1905.11946)
- [Planetary Computer](https://planetarycomputer.microsoft.com/)

## Credits
- Developed by pixelpawnshop

---

## License
MIT License
