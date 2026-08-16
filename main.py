import os
import io
import time
import base64
import traceback
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

app = FastAPI(
    title="Smart Campus Waste Classifier — High-Precision YOLO Microservice",
    description="Production FastAPI computer-vision microservice for waste classification & person safety checks powered by Ultralytics YOLO.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ENVIRONMENT CONFIGURATION VARIABLES
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "weights/best.pt")
if not os.path.exists(MODEL_PATH):
    # Fallback to local pretrained checkpoint if custom trained weights not present
    MODEL_PATH = "yolov8n.pt"

WASTE_CONF_THRESHOLD = float(os.getenv("WASTE_CONFIDENCE_THRESHOLD", "0.65"))
HIGH_CONF_THRESHOLD = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.75"))
PERSON_CONF_THRESHOLD = float(os.getenv("PERSON_CONFIDENCE_THRESHOLD", "0.35"))
YOLO_IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", "640"))
YOLO_IOU_THRESHOLD = float(os.getenv("YOLO_IOU_THRESHOLD", "0.45"))

# MODEL STARTUP LOAD (Loaded ONCE at application startup)
yolo_model = None
model_identifier = "YOLOv8-Standard"

if YOLO_AVAILABLE:
    try:
        if os.path.exists(MODEL_PATH):
            yolo_model = YOLO(MODEL_PATH)
            model_identifier = f"Custom-Waste-YOLO ({Path(MODEL_PATH).name})"
            print(f"[STARTUP] Loaded custom waste model weights from: {MODEL_PATH}")
        else:
            yolo_model = YOLO("yolov8n.pt")
            model_identifier = "YOLOv8n-COCO-Base"
            print("[STARTUP] Loaded fallback YOLOv8n checkpoint.")
    except Exception as e:
        print(f"[STARTUP ERROR] Could not load YOLO model: {e}")

# FINE-GRAINED VISUAL MODEL CLASS -> BIN CATEGORY MAPPING
CLASS_TO_CATEGORY_MAP: Dict[str, str] = {
    # 1. Recyclables (Plastics, Metals, Containers)
    "plastic_bottle": "Recyclable",
    "plastic_container": "Recyclable",
    "plastic_cup": "Recyclable",
    "plastic_bag": "Recyclable",
    "aluminum_can": "Recyclable",
    "metal_can": "Recyclable",
    "bottle": "Recyclable",
    "cup": "Recyclable",
    "can": "Recyclable",
    "bowl": "Recyclable",
    "spoon": "Recyclable",
    "fork": "Recyclable",
    "knife": "Recyclable",

    # 2. Paper & Cardboard
    "paper": "Paper",
    "cardboard": "Paper",
    "newspaper": "Paper",
    "book": "Paper",
    "box": "Paper",

    # 3. Organic & Food Waste
    "food_waste": "Organic",
    "fruit_waste": "Organic",
    "vegetable_waste": "Organic",
    "apple": "Organic",
    "banana": "Organic",
    "orange": "Organic",
    "broccoli": "Organic",
    "carrot": "Organic",
    "sandwich": "Organic",
    "donut": "Organic",
    "cake": "Organic",

    # 4. Glass
    "glass_bottle": "Glass",
    "glass_jar": "Glass",
    "wine glass": "Glass",
    "glass": "Glass",

    # 5. E-Waste
    "battery": "E-Waste",
    "mobile_phone": "E-Waste",
    "electronic_waste": "E-Waste",
    "cell phone": "E-Waste",
    "laptop": "E-Waste",
    "keyboard": "E-Waste",
    "mouse": "E-Waste",
    "tv": "E-Waste",
    "remote": "E-Waste",

    # 6. Non-Recyclable / Miscellaneous
    "textile": "Non-Recyclable",
    "sanitary_waste": "Non-Recyclable",
    "non_recyclable": "Non-Recyclable",
    "other_waste": "Non-Recyclable",
    "trash": "Non-Recyclable"
}

def validate_image_quality(image_np: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Validates image brightness and blurriness before running inference.
    Returns rejection dict if quality check fails, else None.
    """
    # 1. Brightness Check
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if CV2_AVAILABLE else np.mean(image_np, axis=2).astype(np.uint8)
    mean_brightness = float(np.mean(gray))

    if mean_brightness < 30.0:
        return {
            "success": True,
            "is_waste": False,
            "reason": "poor_lighting_dark",
            "message": "The image is too dark to identify the waste item. Please turn on lights or move to a brighter spot.",
            "detections": []
        }
    if mean_brightness > 240.0:
        return {
            "success": True,
            "is_waste": False,
            "reason": "poor_lighting_overexposed",
            "message": "The image is severely overexposed. Please avoid direct harsh camera glare and retake.",
            "detections": []
        }

    # 2. Blurriness Check (Laplacian Variance)
    if CV2_AVAILABLE:
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_var < 80.0:
            return {
                "success": True,
                "is_waste": False,
                "reason": "image_blurry",
                "message": "The image is too blurry. Please hold the camera steady and retake the photo.",
                "detections": []
            }

    return None

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "High-Precision Waste AI Computer-Vision Pipeline",
        "yolo_loaded": yolo_model is not None,
        "model_identifier": model_identifier,
        "config": {
            "waste_confidence_threshold": WASTE_CONF_THRESHOLD,
            "person_confidence_threshold": PERSON_CONF_THRESHOLD,
            "image_size": YOLO_IMAGE_SIZE
        }
    }

@app.post("/classify")
async def classify_image(request: Request):
    t_start = time.perf_counter()
    try:
        content_type = request.headers.get("content-type", "").lower()
        image_bytes = None

        if "multipart/form-data" in content_type:
            form = await request.form()
            file_obj = form.get("file")
            if file_obj and hasattr(file_obj, "read"):
                image_bytes = await file_obj.read()
        else:
            try:
                body = await request.json()
            except Exception:
                body = {}
            b64_str = body.get("imageBase64", "")
            if b64_str:
                if "," in b64_str:
                    b64_str = b64_str.split(",")[1]
                b64_str = b64_str.replace(" ", "+")
                missing_padding = len(b64_str) % 4
                if missing_padding:
                    b64_str += "=" * (4 - missing_padding)
                try:
                    image_bytes = base64.b64decode(b64_str)
                except Exception as b64_err:
                    print(f"Base64 decode error: {b64_err}")

        if not image_bytes:
            t_end = time.perf_counter()
            return {
                "success": False,
                "is_waste": False,
                "reason": "invalid_payload",
                "message": "No valid image payload provided in request.",
                "detections": [],
                "inference_time_ms": round((t_end - t_start) * 1000, 2)
            }

        # Convert to PIL RGB & Numpy
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(pil_img)

        # 1. Image Quality Check
        quality_rejection = validate_image_quality(img_np)
        if quality_rejection:
            t_end = time.perf_counter()
            quality_rejection["inference_time_ms"] = round((t_end - t_start) * 1000, 2)
            return quality_rejection

        # 2. YOLO Object Detection Inference
        all_detections = []
        person_detections = []
        waste_detections = []

        if yolo_model is not None:
            results = yolo_model(pil_img, imgsz=YOLO_IMAGE_SIZE, conf=0.15, iou=YOLO_IOU_THRESHOLD, verbose=False)
            for r in results:
                if hasattr(r, "boxes") and r.boxes is not None:
                    img_h, img_w = r.orig_shape if hasattr(r, "orig_shape") else (pil_img.height, pil_img.width)
                    for box in r.boxes:
                        cls_id = int(box.cls[0].item() if hasattr(box.cls[0], "item") else box.cls[0])
                        cls_name = r.names[cls_id].lower() if (r.names and cls_id in r.names) else "object"
                        conf = float(box.conf[0].item() if hasattr(box.conf[0], "item") else box.conf[0])
                        xyxy = [float(x) for x in box.xyxy[0].tolist()]

                        mapped_category = CLASS_TO_CATEGORY_MAP.get(cls_name, "Recyclable")

                        det_item = {
                            "class_name": cls_name,
                            "confidence": round(conf, 4),
                            "bbox": [round(xyxy[0], 1), round(xyxy[1], 1), round(xyxy[2], 1), round(xyxy[3], 1)],
                            "normalized_bbox": [
                                round(xyxy[0] / img_w, 4),
                                round(xyxy[1] / img_h, 4),
                                round(xyxy[2] / img_w, 4),
                                round(xyxy[3] / img_h, 4)
                            ],
                            "category": mapped_category
                        }
                        all_detections.append(det_item)

                        if cls_name == "person":
                            person_detections.append(det_item)
                        else:
                            waste_detections.append(det_item)

        t_end = time.perf_counter()
        inference_time_ms = round((t_end - t_start) * 1000, 2)

        # 3. HUMAN DETECTION HARD SAFETY CHECK
        # Rejects image if a person is detected with confidence >= PERSON_CONF_THRESHOLD (0.35)
        high_conf_persons = [p for p in person_detections if p["confidence"] >= PERSON_CONF_THRESHOLD]
        if high_conf_persons:
            top_person = max(high_conf_persons, key=lambda x: x["confidence"])
            return {
                "success": True,
                "is_waste": False,
                "reason": "person_detected",
                "message": "This image contains a person. Please point the camera directly at the waste item.",
                "confidence": round(top_person["confidence"] * 100, 1),
                "detections": all_detections,
                "model": model_identifier,
                "inference_time_ms": inference_time_ms
            }

        # 4. WASTE DETECTION & CONFIDENCE FILTERING
        # Filter waste objects meeting minimum confidence threshold (0.65)
        valid_waste = [w for w in waste_detections if w["confidence"] >= WASTE_CONF_THRESHOLD]

        if not valid_waste:
            return {
                "success": True,
                "is_waste": False,
                "reason": "low_confidence",
                "message": "Unable to confidently identify this waste item. Please move closer, improve lighting, and center the item.",
                "detections": all_detections,
                "model": model_identifier,
                "inference_time_ms": inference_time_ms
            }

        # Select primary (highest confidence) waste object
        valid_waste.sort(key=lambda d: d["confidence"], reverse=True)
        primary = valid_waste[0]
        raw_class = primary["class_name"]
        category = primary["category"]
        conf_pct = round(primary["confidence"] * 100, 1)

        display_name = raw_class.replace("_", " ").title()

        return {
            "success": True,
            "is_waste": True,
            "reason": None,
            "message": f"Identified valid waste object '{display_name}' ({conf_pct}% confidence).",
            "primary_detection": primary,
            "category": category,
            "confidence": conf_pct,
            "item_name": f"Identified Item: {display_name}",
            "description": f"AI localized '{display_name}' with {conf_pct}% confidence.",
            "recommended_bin_category": category if category != "Glass" else "Recyclable",
            "detections": valid_waste,
            "model": model_identifier,
            "inference_time_ms": inference_time_ms
        }

    except Exception as e:
        print(f"[CLASSIFY INTERNAL ERROR] {e}")
        traceback.print_exc()
        t_end = time.perf_counter()
        return {
            "success": False,
            "is_waste": False,
            "reason": "service_error",
            "message": f"AI classification error: {str(e)}",
            "detections": [],
            "inference_time_ms": round((t_end - t_start) * 1000, 2)
        }
