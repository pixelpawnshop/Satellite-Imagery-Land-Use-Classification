# Satellite Imagery Land Use Classification (EuroSAT)

## Project Structure

- `data/` - Raw and processed datasets
- `models/` - Saved model weights and architectures
- `notebooks/` - Jupyter notebooks for exploration and prototyping
- `scripts/` - Python scripts for data processing, training, evaluation
- `serving/` - FastAPI app and Docker files for model serving
- `experiments/` - MLflow/W&B logs and experiment tracking
- `visualization/` - Streamlit dashboards, plots, and visual outputs
- `configs/` - Configuration files (YAML/JSON)

## Tech Stack
- Python, PyTorch
- FastAPI, Docker
- SQLite/PostgreSQL
- MLflow/W&B
- Streamlit (optional)

## Getting Started
1. Clone the repo
2. Install dependencies
3. Download EuroSAT dataset
4. Run preprocessing and training scripts
5. Serve model via FastAPI
6. Visualize results

---

This project demonstrates MLOps best practices for satellite imagery land use classification using transfer learning and PyTorch.