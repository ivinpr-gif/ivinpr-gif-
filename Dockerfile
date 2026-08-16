FROM python:3.10-slim

# Install system dependencies for OpenCV, PyTorch, and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and scripts
COPY main.py .
COPY dataset_utils.py .
COPY train.py .
COPY benchmark.py .

# Copy model weights if present
COPY yolov8n.pt .

# Pre-warm PyTorch YOLO model cache during build
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

EXPOSE 8000

# Support dynamic PORT environment variable assigned by Render/Railway cloud platforms
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
