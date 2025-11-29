# EuroSAT Land Use Classification - FastAPI Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY ./serving ./serving
COPY ./models ./models
COPY ./scripts ./scripts
COPY ./visualization ./visualization
COPY ./reports ./reports

# Expose port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
