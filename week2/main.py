# app.py
# pip install -r requirements.txt
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Tuple
from PIL import Image, ImageOps
import numpy as np
import cv2
import io
import base64
import os

app = FastAPI(title="Leaf Health Checker", version="1.1")

# ----------------------------
# Tunables (override via env)
# ----------------------------
GREEN_LO = np.array([int(os.getenv("GREEN_H_LO", 25)),
                     int(os.getenv("GREEN_S_LO", 40)),
                     int(os.getenv("GREEN_V_LO", 20))])
GREEN_HI = np.array([int(os.getenv("GREEN_H_HI", 95)),
                     int(os.getenv("GREEN_S_HI", 255)),
                     int(os.getenv("GREEN_V_HI", 255))])

YB1_LO = np.array([int(os.getenv("YB_H_LO", 5)), 30, 20])
YB1_HI = np.array([int(os.getenv("YB_H_HI", 35)), 255, 255])

LOW_SAT_MAX = int(os.getenv("LOW_SAT_MAX", 50))
DARK_V_MAX = int(os.getenv("DARK_V_MAX", 80))

EDGE_CANNY1 = int(os.getenv("EDGE_CANNY1", 60))
EDGE_CANNY2 = int(os.getenv("EDGE_CANNY2", 120))

LEAF_MIN_AREA = int(os.getenv("LEAF_MIN_AREA", 2000))
BLUR_VAR_MIN = float(os.getenv("BLUR_VAR_MIN", 60.0))
V_LOW = int(os.getenv("V_LOW", 60))
V_HIGH = int(os.getenv("V_HIGH", 215))

W_YB = float(os.getenv("W_YB", 0.6))
W_DL = float(os.getenv("W_DL", 0.3))
W_EDGE = float(os.getenv("W_EDGE", 0.1))

# ----------------------------
# Models
# ----------------------------
class PredictResponse(BaseModel):
    category: str
    health_percentage: int
    confidence: float
    signals: dict
    first_aid: List[str]
    quality_warnings: List[str] = []
    overlay_png_b64: Optional[str] = None


# ----------------------------
# Utils
# ----------------------------
def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def preprocess(img: Image.Image, out_size=512) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((out_size, out_size), Image.BICUBIC)
    return img

def gray_world_white_balance(bgr: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(bgr.astype(np.float32))
    mean_b, mean_g, mean_r = b.mean(), g.mean(), r.mean()
    mean_gray = (mean_b + mean_g + mean_r) / 3.0 + 1e-6
    b *= (mean_gray / mean_b)
    g *= (mean_gray / mean_g)
    r *= (mean_gray / mean_r)
    out = cv2.merge([b, g, r])
    return np.clip(out, 0, 255).astype(np.uint8)

def lab_clahe(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def quality_checks(bgr: np.ndarray, hsv: np.ndarray, leaf_mask: np.ndarray) -> Tuple[List[str], float]:
    warnings = []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    v_mean = hsv[..., 2].mean()
    v_std = hsv[..., 2].std()
    leaf_area = int((leaf_mask > 0).sum())

    if blur_var < BLUR_VAR_MIN:
        warnings.append("Image appears blurry; retake with steady hand or better focus.")
    if v_mean < V_LOW:
        warnings.append("Image underexposed; take photo in brighter, indirect light.")
    if v_mean > V_HIGH:
        warnings.append("Image overexposed; avoid harsh sun or reduce exposure.")
    if leaf_area < LEAF_MIN_AREA:
        warnings.append("Leaf too small in frame; move closer and fill the image.")

    # Simple confidence: combine normalized blur, exposure, and leaf area
    # Normalize terms to 0..1 and clamp
    blur_score = np.clip(blur_var / max(BLUR_VAR_MIN, 1.0), 0.0, 1.5)
    expo_penalty = 1.0 - (abs(v_mean - 128) / 128.0)
    area_score = np.clip(leaf_area / (512 * 512 * 0.4), 0.0, 1.0)
    confidence = float(np.clip(0.25 * blur_score + 0.35 * expo_penalty + 0.40 * area_score, 0.0, 1.0))
    return warnings, confidence

def kmeans_leaf_fallback(bgr: np.ndarray) -> np.ndarray:
    # Fallback segmentation if HSV thresholding fails
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    Z = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    K = 3
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(img.shape[:2])
    # Assume leaf is the cluster with highest green channel in original BGR
    masks = []
    for k in range(K):
        mask = (labels == k).astype(np.uint8) * 255
        g_mean = cv2.mean(cv2.cvtColor(cv2.bitwise_and(bgr, bgr, mask=mask), cv2.COLOR_BGR2RGB))[1]
        masks.append((g_mean, mask))
    masks.sort(key=lambda x: x[0], reverse=True)
    leaf_mask = masks[0][1]
    kernel = np.ones((5, 5), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return leaf_mask

def leaf_mask_and_damage(bgr: np.ndarray):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Primary leaf mask
    leaf_mask = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    kernel = np.ones((5, 5), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    leaf_area = int((leaf_mask > 0).sum())

    # Fallback if too small
    if leaf_area < LEAF_MIN_AREA:
        leaf_mask = kmeans_leaf_fallback(bgr)
        leaf_area = int((leaf_mask > 0).sum())

    if leaf_area < LEAF_MIN_AREA:
        return hsv, leaf_mask, 1.0, {"note": "Leaf not clearly detected"}, {"yb":0,"dl":0,"edge":0}

    # Yellow/brown
    yb1 = cv2.inRange(hsv, YB1_LO, YB1_HI)
    lowS = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, LOW_SAT_MAX, 255]))
    yb_mask = cv2.bitwise_or(yb1, lowS)
    yb_mask = cv2.bitwise_and(yb_mask, leaf_mask)

    # Dark lesions
    dl_mask = cv2.inRange(hsv, np.array([0, 40, 0]), np.array([180, 255, DARK_V_MAX]))
    dl_mask = cv2.bitwise_and(dl_mask, leaf_mask)

    # Edge density for chew/holes
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, EDGE_CANNY1, EDGE_CANNY2)
    edges = cv2.bitwise_and(edges, edges, mask=leaf_mask)
    edge_ratio = edges.sum() / (leaf_area * 255.0)

    yb_area = int((yb_mask > 0).sum())
    dl_area = int((dl_mask > 0).sum())

    damage_pixels = W_YB * yb_area + W_DL * dl_area + W_EDGE * (edge_ratio * leaf_area)
    damage_ratio = float(np.clip(damage_pixels / max(leaf_area, 1), 0.0, 1.0))

    cues = {
        "yellowing_share": yb_area / max(leaf_area, 1),
        "dark_lesion_share": dl_area / max(leaf_area, 1),
        "edge_density": float(edge_ratio),
        "note": None
    }
    masks = {"yb": yb_mask, "dl": dl_mask, "edge": edges}
    return hsv, leaf_mask, damage_ratio, cues, masks

def categorize(health_pct: int) -> str:
    if health_pct >= 90:
        return "Healthy"
    elif health_pct >= 70:
        return "Mildly Unhealthy"
    return "Severe"

def first_aid(cues: dict) -> List[str]:
    tips = []
    if cues["dark_lesion_share"] > 0.08:
        tips += [
            "Prune heavily spotted leaves; sterilize shears between cuts.",
            "Improve airflow; avoid overhead watering late in the day.",
            "Use a crop-safe fungicide (e.g., copper or bio-fungicide) exactly as labeled."
        ]
    if cues["yellowing_share"] > 0.12:
        tips += [
            "Verify watering schedule: evenly moist, not waterlogged.",
            "Apply a balanced fertilizer; consider a nitrogen boost if older leaves yellow first.",
            "Check soil pH; many crops prefer 6.0–6.8."
        ]
    if cues["edge_density"] > 0.006:
        tips += [
            "Inspect undersides for chewing pests.",
            "Hand-pick or use targeted controls such as neem or BT where appropriate.",
            "Use row covers or traps if pressure persists."
        ]
    if not tips:
        tips = [
            "Remove visibly damaged leaves.",
            "Water in the morning and keep foliage as dry as possible.",
            "Reassess after 48–72 hours."
        ]
    return tips[:5]

def make_overlay_png(bgr: np.ndarray, leaf_mask: np.ndarray, masks: dict) -> str:
    overlay = bgr.copy()
    # Colorize masks for visualization
    yb_c = cv2.applyColorMap((masks["yb"]>0).astype(np.uint8)*255, cv2.COLORMAP_AUTUMN)
    dl_c = cv2.applyColorMap((masks["dl"]>0).astype(np.uint8)*255, cv2.COLORMAP_OCEAN)
    edge_c = cv2.cvtColor(masks["edge"], cv2.COLOR_GRAY2BGR)

    # Blend only within leaf
    leaf_mask_3 = cv2.merge([leaf_mask]*3) // 255
    overlay = overlay * (1 - leaf_mask_3) + (overlay * 0.6 + 0.4 * yb_c) * leaf_mask_3
    overlay = overlay * (1 - leaf_mask_3) + (overlay * 0.7 + 0.3 * dl_c) * leaf_mask_3
    overlay = cv2.addWeighted(overlay.astype(np.uint8), 1.0, edge_c, 0.4, 0)

    # Outline leaf
    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    # Encode PNG -> base64
    ok, buf = cv2.imencode(".png", overlay.astype(np.uint8))
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return b64

# ----------------------------
# Endpoint
# ----------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    include_overlay: bool = Query(False, description="Return overlay PNG as base64")
):
    data = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(data))
    except Exception:
        return JSONResponse({"status": "error", "message": "Unsupported or corrupted image file."}, status_code=400)

    pil_img = preprocess(pil_img, out_size=512)
    bgr = pil_to_bgr(pil_img)
    bgr = gray_world_white_balance(bgr)
    bgr = lab_clahe(bgr)

    hsv, leaf_mask, damage_ratio, cues, masks = leaf_mask_and_damage(bgr)

    if cues.get("note") == "Leaf not clearly detected":
        return JSONResponse({
            "status": "error",
            "message": "Couldn't detect a clear leaf. Please upload a closer, well-lit photo on a plain background."
        }, status_code=200)

    health_pct = int(round(max(0.0, 100.0 - 100.0 * damage_ratio)))
    category = categorize(health_pct)

    warnings, conf = quality_checks(bgr, hsv, leaf_mask)
    tips = first_aid(cues)

    overlay_b64 = make_overlay_png(bgr, leaf_mask, masks) if include_overlay else None

    return {
        "category": category,
        "health_percentage": health_pct,
        "confidence": round(conf, 3),
        "signals": {
            "yellowing_fraction": round(float(cues["yellowing_share"]), 3),
            "lesion_fraction": round(float(cues["dark_lesion_share"]), 3),
            "edge_density": round(float(cues["edge_density"]), 4)
        },
        "first_aid": tips,
        "quality_warnings": warnings,
        "overlay_png_b64": overlay_b64
    }
