# ============================================================
# CAPSICUM PLANT CONDITION DETECTION API
# ResNet50 | Layer4 + FC fine-tuning | 13 classes
# Compatible with train_augu.py: Dropout(0.30) + Linear(2048, 13)
# ============================================================
import io
import os
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models, transforms
from class_info import CLASS_INFO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_SIZE = 224
TRAIN_TARGET = 1100
NUM_CLASSES = 13
EXPECTED_TRAIN_TOTAL = TRAIN_TARGET * NUM_CLASSES
TOP_K = 5
DROPOUT = 0.30
MAX_UPLOAD_MB = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPECTED_CLASSES = [
    "Anthracnose", "Larva", "Magnesium", "Phytophthora blight",
    "bacterial spot", "blossom-end rot", "down leaf aphid", "fruit thrips",
    "healthy", "powdery mildew", "snail", "upperleaf thrips", "virus",
]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
TRAIN_DIR = os.path.join(BASE_DIR, "augmented_dataset", "train")
VALID_DIR = os.path.join(BASE_DIR, "augmented_dataset", "valid")
TEST_DIR = os.path.join(BASE_DIR, "augmented_dataset", "test")


def resolve_model_path():
    env_path = os.environ.get("MODEL_PATH", "").strip()
    if env_path:
        return os.path.abspath(env_path)
    candidates = [
        os.path.join(BASE_DIR, "models", "condition_resnet50_best.pth"),
        os.path.join(BASE_DIR, "models", "condition_resnet50.pth"),
        os.path.join(BASE_DIR, "condition_resnet50_best.pth"),
        os.path.join(BASE_DIR, "condition_resnet50.pth"),
    ]
    return next((p for p in candidates if os.path.isfile(p)), candidates[0])

MODEL_PATH = resolve_model_path()
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
model: Optional[nn.Module] = None
MODEL_CLASSES: List[str] = []
MODEL_ARCHITECTURE = "ResNet50 + Layer4 Fine-Tuning + Dropout(0.30) + Linear(2048,13)"
CHECKPOINT_INFO = {}


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model_state_dict", "state_dict"):
        if isinstance(checkpoint.get(key), dict):
            return checkpoint[key]
    return checkpoint


def clean_state_dict(state_dict):
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}


def get_checkpoint_classes(checkpoint):
    if isinstance(checkpoint, dict):
        classes = checkpoint.get("classes")
        if isinstance(classes, (list, tuple)):
            return list(classes)
        mapping = checkpoint.get("class_to_idx")
        if isinstance(mapping, dict):
            try:
                return [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]
            except (TypeError, ValueError):
                raise RuntimeError("Checkpoint class_to_idx mapping is invalid.")
    return EXPECTED_CLASSES.copy()


def validate_class_order(classes):
    if len(classes) != NUM_CLASSES:
        raise RuntimeError(f"Checkpoint contains {len(classes)} classes; expected {NUM_CLASSES}.")
    if classes != EXPECTED_CLASSES:
        raise RuntimeError(
            "Checkpoint class order does not match app.py.\n"
            f"Expected: {EXPECTED_CLASSES}\nCheckpoint: {classes}"
        )


def build_model(state_dict):
    network = models.resnet50(weights=None)
    if network.fc.in_features != 2048:
        raise RuntimeError(f"Unexpected ResNet50 FC input: {network.fc.in_features}")
    if "fc.1.weight" not in state_dict or "fc.1.bias" not in state_dict:
        keys = [key for key in state_dict if key.startswith("fc.")]
        raise RuntimeError(
            "Checkpoint architecture mismatch: expected Dropout + Linear with fc.1 weights. "
            f"Found FC keys: {keys}"
        )
    if tuple(state_dict["fc.1.weight"].shape) != (NUM_CLASSES, 2048):
        raise RuntimeError(f"Unexpected FC weight shape: {tuple(state_dict['fc.1.weight'].shape)}")
    if tuple(state_dict["fc.1.bias"].shape) != (NUM_CLASSES,):
        raise RuntimeError(f"Unexpected FC bias shape: {tuple(state_dict['fc.1.bias'].shape)}")
    network.fc = nn.Sequential(nn.Dropout(p=DROPOUT), nn.Linear(2048, NUM_CLASSES))
    network.load_state_dict(state_dict, strict=True)
    network.to(DEVICE).eval()
    return network


def load_model():
    global model, MODEL_CLASSES, CHECKPOINT_INFO
    print("\n" + "=" * 70)
    print("LOADING CAPSICUM RESNET50 CHECKPOINT")
    print(f"Model path: {MODEL_PATH}\nDevice: {DEVICE}")
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}\n"
            "Place condition_resnet50_best.pth or condition_resnet50.pth in the models folder."
        )
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    CHECKPOINT_INFO = checkpoint if isinstance(checkpoint, dict) else {}
    state_dict = clean_state_dict(extract_state_dict(checkpoint))
    MODEL_CLASSES = get_checkpoint_classes(checkpoint)
    validate_class_order(MODEL_CLASSES)
    model = build_model(state_dict)
    print("Model loaded successfully:", MODEL_ARCHITECTURE)
    print("=" * 70)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    global model
    model = None

app = FastAPI(
    title="Capsicum Plant Condition Detection API",
    description=(
        "ResNet50 API for 13 Capsicum conditions. Returns predicted class, "
        "softmax confidence estimate, Top-5 alternatives and class information. "
        "Preprocessing: Resize 224x224, ToTensor, ImageNet normalization."
    ),
    version="7.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)


def count_images(directory: str) -> int:
    if not os.path.isdir(directory):
        return 0
    return sum(1 for _, _, files in os.walk(directory)
               for name in files if name.lower().endswith(IMAGE_EXTENSIONS))


def count_images_per_class(directory: str):
    return {name: count_images(os.path.join(directory, name)) for name in EXPECTED_CLASSES}


def safe_class_info(class_name: str):
    if class_name in CLASS_INFO:
        return CLASS_INFO[class_name]
    target = class_name.strip().casefold()
    for key, value in CLASS_INFO.items():
        if key.strip().casefold() == target:
            return value
    return {"information_available": False,
            "message": "No information configured for this class in class_info.py."}


def predict_image(image: Image.Image):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    tensor = INFERENCE_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        probs = torch.softmax(model(tensor), dim=1)[0]
        top_probs, top_indices = torch.topk(probs, k=min(TOP_K, len(MODEL_CLASSES)))
    predictions = []
    for rank, (prob, idx) in enumerate(zip(top_probs.tolist(), top_indices.tolist()), 1):
        name = MODEL_CLASSES[idx]
        predictions.append({
            "rank": rank, "class": name, "class_index": idx,
            "confidence_percent": round(prob * 100.0, 4),
            "class_information": safe_class_info(name),
        })
    primary = predictions[0]
    confidence = primary["confidence_percent"]
    level = "HIGH" if confidence >= 90 else "MEDIUM" if confidence >= 70 else "LOW" if confidence >= 50 else "VERY LOW"
    warning = None if confidence >= 50 else (
        "Low model confidence. Check image clarity/lighting; the image may be outside "
        "the supported classes."
    )
    return {
        "predicted_class": primary["class"],
        "class_index": primary["class_index"],
        "confidence_percent": confidence,
        "confidence_level": level,
        "confidence_note": "Softmax confidence is a model estimate, not a guarantee of correctness.",
        "warning": warning,
        "class_information": primary["class_information"],
        "top_5_predictions": predictions,
    }


async def read_uploaded_image(file: UploadFile) -> Image.Image:
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Upload JPG, JPEG, PNG, WEBP, BMP or TIFF.")
    contents = await file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_UPLOAD_MB} MB limit.")
    try:
        with Image.open(io.BytesIO(contents)) as source:
            source.load()
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")


@app.get("/")
def root():
    return {
        "status": "running", "application": "Capsicum Plant Condition Detection API",
        "model_loaded": model is not None, "model": "ResNet50",
        "architecture": MODEL_ARCHITECTURE, "fine_tuned_layers": ["layer4", "fc"],
        "num_classes": NUM_CLASSES, "classes": MODEL_CLASSES,
        "input_size": "224x224", "device": str(DEVICE), "docs": "/docs",
        "augmentation": {"method": "Adaptive offline augmentation", "applied_to": "training set only",
                          "validation_augmented": False, "test_augmented": False,
                          "runtime_augmentation": False},
        "soil_model_included": False,
    }


@app.get("/health")
def health():
    return {"status": "healthy" if model is not None else "model_not_loaded",
            "cnn_model_loaded": model is not None, "model": "ResNet50",
            "num_classes": len(MODEL_CLASSES), "device": str(DEVICE)}


@app.get("/classes")
def classes():
    return {"num_classes": len(MODEL_CLASSES), "classes": MODEL_CLASSES}


@app.get("/model-info")
def model_info():
    result = {
        "model": "ResNet50", "architecture": MODEL_ARCHITECTURE,
        "framework": "PyTorch", "input_size": "224x224", "input_features": 2048,
        "num_classes": NUM_CLASSES, "classes": MODEL_CLASSES,
        "fine_tuned_layers": ["layer4", "fc"], "dropout": DROPOUT,
        "device": str(DEVICE), "model_path": MODEL_PATH,
        "checkpoint_exists": os.path.isfile(MODEL_PATH),
        "training_data": {"train": TRAIN_DIR, "validation": VALID_DIR, "test": TEST_DIR},
        "preprocessing": ["Resize(224,224)", "ToTensor()", "ImageNet normalization"],
    }
    for key in ("epoch", "best_train_accuracy", "best_valid_accuracy", "architecture", "num_classes"):
        if key in CHECKPOINT_INFO:
            output_key = "best_model_epoch" if key == "epoch" else key
            value = CHECKPOINT_INFO[key]
            if key == "epoch" and isinstance(value, int):
                value += 1
            result[output_key] = value
    return result


@app.get("/class-info/{class_name}")
def get_class_info(class_name: str):
    target = class_name.strip().casefold()
    for name, info in CLASS_INFO.items():
        if name.strip().casefold() == target:
            return {"class": name, "information": info}
    raise HTTPException(status_code=404, detail={
        "message": "Class information not found", "requested_class": class_name,
        "available_classes": list(CLASS_INFO.keys()),
    })


@app.get("/dataset")
def dataset_info():
    train_counts = count_images_per_class(TRAIN_DIR)
    train_count, valid_count, test_count = count_images(TRAIN_DIR), count_images(VALID_DIR), count_images(TEST_DIR)
    return {
        "train_images": train_count, "validation_images": valid_count, "test_images": test_count,
        "total_images": train_count + valid_count + test_count, "num_classes": NUM_CLASSES,
        "train_target_per_class": TRAIN_TARGET, "expected_train_total": EXPECTED_TRAIN_TOTAL,
        "train_total_correct": train_count == EXPECTED_TRAIN_TOTAL,
        "all_train_classes_1100": all(v == TRAIN_TARGET for v in train_counts.values()),
        "class_counts": train_counts, "train_directory": TRAIN_DIR,
        "validation_directory": VALID_DIR, "test_directory": TEST_DIR,
        "augmentation": "Adaptive offline augmentation; training only",
        "validation_test_augmented": False,
    }


@app.get("/dataset/class-wise")
def dataset_class_wise():
    train, valid, test = (count_images_per_class(p) for p in (TRAIN_DIR, VALID_DIR, TEST_DIR))
    return {"num_classes": NUM_CLASSES, "train_target_per_class": TRAIN_TARGET,
            "expected_train_total": EXPECTED_TRAIN_TOTAL,
            "classes": {name: {"train": train[name], "validation": valid[name], "test": test[name],
                               "train_target": TRAIN_TARGET,
                               "train_target_reached": train[name] == TRAIN_TARGET}
                        for name in MODEL_CLASSES}}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = await read_uploaded_image(file)
    prediction = predict_image(image)
    return {"success": True, "filename": file.filename,
            "image_width": image.width, "image_height": image.height,
            "model": "ResNet50", "architecture": MODEL_ARCHITECTURE,
            "fine_tuned_layers": ["layer4", "fc"], "num_classes": NUM_CLASSES,
            "device": str(DEVICE), "prediction": prediction}


@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No images were uploaded.")
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Upload a maximum of 50 images per batch.")
    results = []
    for file in files:
        try:
            image = await read_uploaded_image(file)
            results.append({"filename": file.filename, "image_width": image.width,
                            "image_height": image.height, "success": True,
                            "prediction": predict_image(image)})
        except HTTPException as exc:
            results.append({"filename": file.filename, "success": False,
                            "error": exc.detail, "status_code": exc.status_code})
        except Exception as exc:
            results.append({"filename": file.filename, "success": False, "error": str(exc)})
    succeeded = sum(item["success"] for item in results)
    return {"success": succeeded == len(results), "total_images": len(results),
            "successful_predictions": succeeded, "failed_predictions": len(results) - succeeded,
            "results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), reload=False)
