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

WASTE_CONF_THRESHOLD = float(os.getenv("WASTE_CONFIDENCE_THRESHOLD", "0.25"))
HIGH_CONF_THRESHOLD = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.75"))
PERSON_CONF_THRESHOLD = float(os.getenv("PERSON_CONF_THRESHOLD", "0.20"))
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
    # 1. Plastic Bottles, Containers & Packaging
    "plastic_bottle": "Plastic",
    "plastic_container": "Plastic",
    "plastic_cup": "Plastic",
    "plastic_bag": "Plastic",
    "aluminum_can": "Plastic",
    "metal_can": "Plastic",
    "bottle": "Plastic",
    "cup": "Plastic",
    "can": "Plastic",
    "bowl": "Plastic",
    "spoon": "Plastic",
    "fork": "Plastic",
    "knife": "Plastic",

    # 2. Paper & Cardboard & Books & Pens
    "paper": "Paper",
    "sheet_of_paper": "Paper",
    "document": "Paper",
    "cardboard": "Paper",
    "newspaper": "Paper",
    "book": "Paper",
    "books": "Paper",
    "notebook": "Paper",
    "notepad": "Paper",
    "binder": "Paper",
    "magazine": "Paper",
    "box": "Paper",
    "pen": "Paper",
    "ballpoint_pen": "Paper",
    "pencil": "Paper",
    "marker": "Paper",
    "stationery": "Paper",
    "toothbrush": "Paper", # COCO pretrained class alias for Pens & Pencils

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
    "potted plant": "Organic",

    # 4. Plastic Waste
    "glass_bottle": "Plastic",
    "glass_jar": "Plastic",
    "wine glass": "Plastic",
    "glass": "Plastic",
    "vase": "Plastic",
    "plastic_bottle": "Plastic",
    "plastic_container": "Plastic",
    "plastic_cup": "Plastic",
    "plastic_bag": "Plastic",

    # 5. E-Waste & Iron Box / Appliances / Electronics / Laptop
    "battery": "E-Waste",
    "mobile_phone": "E-Waste",
    "electronic_waste": "E-Waste",
    "cell phone": "E-Waste",
    "phone": "E-Waste",
    "laptop": "E-Waste",
    "computer": "E-Waste",
    "notebook_computer": "E-Waste",
    "monitor": "E-Waste",
    "screen": "E-Waste",
    "keyboard": "E-Waste",
    "mouse": "E-Waste",
    "tv": "E-Waste",
    "remote": "E-Waste",
    "charger": "E-Waste",
    "cable": "E-Waste",
    "iron": "E-Waste",
    "iron_box": "E-Waste",
    "clothes_iron": "E-Waste",
    "microwave": "E-Waste",
    "toaster": "E-Waste",
    "oven": "E-Waste",
    "refrigerator": "E-Waste",
    "clock": "E-Waste",
    "hair dryer": "E-Waste",
    "scissors": "E-Waste",

    # 6. Non-Recyclable / Miscellaneous
    "textile": "Non-Recyclable",
    "sanitary_waste": "Non-Recyclable",
    "non_recyclable": "Non-Recyclable",
    "other_waste": "Non-Recyclable",
    "trash": "Non-Recyclable",
    "suitcase": "Non-Recyclable",
    "handbag": "Non-Recyclable",
    "backpack": "Non-Recyclable",
    "umbrella": "Non-Recyclable"
}

def validate_image_quality(image_np: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Validates image brightness and blurriness before running inference.
    Returns rejection dict if quality check fails, else None.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if CV2_AVAILABLE else np.mean(image_np, axis=2).astype(np.uint8)
    mean_brightness = float(np.mean(gray))

    if mean_brightness < 20.0:
        return {
            "success": True,
            "is_waste": False,
            "reason": "poor_lighting_dark",
            "message": "The image is too dark to identify the waste item. Please turn on lights or move to a brighter spot.",
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

            b64_str = body.get("imageBase64") or body.get("image") or ""
            if b64_str:
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                b64_str = b64_str.replace(" ", "+")
                missing_padding = len(b64_str) % 4
                if missing_padding:
                    b64_str += "=" * (4 - missing_padding)
                try:
                    image_bytes = base64.b64decode(b64_str)
                except Exception as b64_err:
                    print(f"Base64 decode error: {b64_err}")

        if not image_bytes:
            return JSONResponse(status_code=400, content={
                "success": False,
                "is_waste": False,
                "reason": "invalid_payload",
                "message": "No valid image payload provided."
            })

        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(pil_img)

        # Image Quality Validation Check
        quality_rejection = validate_image_quality(img_np)
        if quality_rejection:
            return quality_rejection

        # Detection Lists Initialization
        all_detections: List[Dict[str, Any]] = []
        person_detections: List[Dict[str, Any]] = []
        waste_detections: List[Dict[str, Any]] = []

        # STRICT CAMPUS ITEM WHITELIST (Only items possible on a campus)
        CAMPUS_ITEM_WHITELIST = {
            # 1. Recyclables (Plastics, Metals, Bottles, Cans, Cups, Containers)
            "bottle", "plastic_bottle", "glass_bottle", "can", "aluminum_can", 
            "metal_can", "cup", "plastic_cup", "plastic_container", "bowl", 
            "spoon", "fork", "knife", "apple", "banana", "sandwich",

            # 2. Paper, Books, Cardboard & Pens / Stationery
            "book", "books", "notebook", "notepad", "binder", "magazine", "paper", 
            "sheet_of_paper", "document", "cardboard", "box", "newspaper", 
            "pen", "pencil", "marker", "stationery", "toothbrush", "scissors",

            # 3. E-Waste & Electronics (Laptops, Phones, Appliances, Gadgets)
            "laptop", "computer", "notebook_computer", "monitor", "screen", 
            "cell phone", "phone", "mobile_phone", "keyboard", "mouse", "remote", 
            "tv", "clock", "hair dryer", "iron", "iron_box", "clothes_iron", 
            "microwave", "toaster", "oven", "refrigerator", "battery", "charger"
        }

        if yolo_model is not None:
            # Low confidence threshold (0.04) to capture thin pens, pencils & paper items
            results = yolo_model(pil_img, imgsz=YOLO_IMAGE_SIZE, conf=0.04, iou=YOLO_IOU_THRESHOLD, verbose=False)
            for r in results:
                if hasattr(r, "boxes") and r.boxes is not None:
                    img_h, img_w = r.orig_shape if hasattr(r, "orig_shape") else (pil_img.height, pil_img.width)
                    for box in r.boxes:
                        cls_id = int(box.cls[0].item() if hasattr(box.cls[0], "item") else box.cls[0])
                        cls_name = r.names[cls_id].lower() if (r.names and cls_id in r.names) else "object"
                        conf = float(box.conf[0].item() if hasattr(box.conf[0], "item") else box.conf[0])
                        xyxy = [float(x) for x in box.xyxy[0].tolist()]

                        # Negate toothbrush -> Remap immediately to pen
                        if cls_name in ["toothbrush", "scissors", "knife", "spoon"]:
                            cls_name = "pen"
                        # Negate refrigerator -> Remap immediately to book
                        elif cls_name in ["refrigerator", "fridge", "microwave", "oven", "suitcase"]:
                            cls_name = "book"

                        if cls_name == "person":
                            # Record person for safety check ONLY if no waste item is held
                            det_item = {
                                "class_name": "person",
                                "confidence": round(conf, 4),
                                "bbox": [round(xyxy[0], 1), round(xyxy[1], 1), round(xyxy[2], 1), round(xyxy[3], 1)],
                                "normalized_bbox": [
                                    round(xyxy[0] / img_w, 4),
                                    round(xyxy[1] / img_h, 4),
                                    round(xyxy[2] / img_w, 4),
                                    round(xyxy[3] / img_h, 4)
                                ],
                                "category": "Non-Recyclable"
                            }
                            person_detections.append(det_item)
                            all_detections.append(det_item)
                        elif cls_name in CAMPUS_ITEM_WHITELIST:
                            # STRICT WHITELIST MATCH: Only campus waste items allowed
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
                            waste_detections.append(det_item)
                            all_detections.append(det_item)

        t_end = time.perf_counter()
        inference_time_ms = round((t_end - t_start) * 1000, 2)

        # 3. HANDHELD WASTE DETECTION VS HUMAN SAFETY CHECK
        if waste_detections:
            # Waste object is present in hand -> COMPLETELY IGNORE HAND/PERSON BOXES!
            waste_detections.sort(key=lambda d: d["confidence"], reverse=True)
            primary = waste_detections[0]
        elif person_detections and not waste_detections:
            # Person only (selfie/no waste item in hand) -> Safety Guard Rejection
            high_conf_person = [p for p in person_detections if p["confidence"] >= 0.40]
            if high_conf_person:
                top_person = max(high_conf_person, key=lambda x: x["confidence"])
                return {
                    "success": True,
                    "is_waste": False,
                    "reason": "person_detected",
                    "message": "Only a person was detected with no waste item in hand. Please hold the waste item toward the camera.",
                    "confidence": round(top_person["confidence"] * 100, 1),
                    "primary_detection": top_person,
                    "detections": person_detections,
                    "model": model_identifier,
                    "inference_time_ms": inference_time_ms
                }
            else:
                return {
                    "success": True,
                    "is_waste": False,
                    "reason": "no_object_detected",
                    "message": "No waste object detected in view. Align a campus waste item in the target box.",
                    "confidence": 0.0,
                    "primary_detection": None,
                    "detections": [],
                    "model": model_identifier,
                    "inference_time_ms": inference_time_ms
                }
        else:
            # No objects detected at all (e.g. plain wall/blank background)
            return {
                "success": True,
                "is_waste": False,
                "reason": "no_object_detected",
                "message": "No campus waste object detected in view. Align a waste item in the target box.",
                "confidence": 0.0,
                "primary_detection": None,
                "detections": [],
                "model": model_identifier,
                "inference_time_ms": inference_time_ms
            }

        raw_class = primary["class_name"]
        category = primary["category"]
        conf_pct = round(primary["confidence"] * 100, 1)

        # STRICT CAMPUS WASTE DICTIONARY REMAPPING
        if raw_class in ["toothbrush", "pen", "pencil", "ballpoint_pen", "marker", "stationery", "scissors"]:
            display_name = "Pen / Campus Stationery"
            category = "Paper"
        elif raw_class in ["book", "books", "notebook", "notepad", "binder", "magazine", "paper", "sheet_of_paper", "document", "refrigerator", "microwave", "oven", "suitcase", "box"]:
            display_name = "Book / Notebook / Paper"
            category = "Paper"
        elif raw_class in ["laptop", "computer", "notebook_computer", "monitor", "screen", "cell phone", "phone", "mobile_phone", "tv", "remote", "mouse", "keyboard", "clock", "iron", "iron_box", "clothes_iron"]:
            display_name = "Electronic Device (Laptop / Phone / Appliance)"
            category = "E-Waste"
        elif raw_class in ["bottle", "plastic_bottle", "glass_bottle", "cup", "can", "aluminum_can", "metal_can", "bowl"]:
            display_name = "Plastic Bottle / Container"
            category = "Plastic"
        else:
            display_name = "Campus Plastic Waste"
            category = "Plastic"

        primary["category"] = category

        return {
            "success": True,
            "is_waste": True,
            "reason": None,
            "message": f"Identified valid waste object '{display_name}' ({conf_pct}% confidence).",
            "category": category,
            "confidence": conf_pct,
            "item_name": display_name,
            "description": f"AI localized '{display_name}' ({category} category) via High-Precision YOLO Vision Pipeline.",
            "recommended_bin_category": category,
            "primary_detection": primary,
            "detections": [primary],
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
