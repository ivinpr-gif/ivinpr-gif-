# YOLOv8 FastAPI Inference Microservice

Two-stage object detection microservice for **Smart Campus Waste Disposal & EcoCredit System** powered by `ultralytics` YOLOv8.

---

## 🚀 Local Run Instructions

### 1. Install Dependencies
```bash
cd yolo_service
pip install -r requirements.txt
```

### 2. Start Uvicorn Server
```bash
uvicorn main:app --reload --port 8000
```
API endpoint is accessible at `http://localhost:8000/classify` and interactive Swagger docs at `http://localhost:8000/docs`.

---

## 🐳 Docker / Hugging Face Spaces Deployment

### Deploy to Hugging Face Spaces (Docker SDK)
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces) select **Docker** SDK.
2. Push `Dockerfile`, `main.py`, and `requirements.txt` to the repository.
3. Once built, copy the public Space URL (e.g. `https://your-space.hf.space/classify`) and set it in your environment:
   `YOLO_INFERENCE_SERVICE_URL=https://your-space.hf.space/classify`

---

## 📡 API Endpoint Reference

### `POST /classify`
**Payload (Base64 JSON):**
```json
{
  "imageBase64": "data:image/jpeg;base64,...",
  "imageName": "waste_photo.jpg"
}
```

**Response (Waste Item Detected):**
```json
{
  "is_waste": true,
  "category": "Recyclable",
  "confidence": 94.2,
  "item_name": "YOLOv8 Object: Bottle",
  "description": "YOLOv8 detected 'bottle' with bounding box [60, 40, 180, 220].",
  "detections": [
    { "class": "bottle", "confidence": 0.942, "bbox": [60, 40, 180, 220] }
  ],
  "mapped_category": "Recyclable"
}
```

**Response (Person / Human Rejected):**
```json
{
  "is_waste": false,
  "reason": "person_detected",
  "message": "This looks like a person, not a waste item. Please photograph the item you want to dispose of.",
  "confidence": 99.1
}
```
