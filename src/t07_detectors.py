"""Local pretrained detectors and explicit synthetic-image proposal heuristics.

No network/download code is used at inference. Supplemental proposals are NOT
a trained head detector and make no claim of generalisation to real faces.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from .common.io import sha256_file


def clip_box(box: dict, width: int, height: int, padding: float = 0.0) -> dict:
    values = [float(box[k]) for k in ("x", "y", "w", "h")]
    if not all(math.isfinite(v) for v in values) or values[2] <= 0 or values[3] <= 0:
        raise ValueError("Bounding box must have finite coordinates and positive size")
    x, y, w, h = values
    x0, y0 = max(0, math.floor(x - padding*w)), max(0, math.floor(y - padding*h))
    x1, y1 = min(width, math.ceil(x + w + padding*w)), min(height, math.ceil(y + h + padding*h))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Bounding box is outside image")
    return {**box, "x": x0, "y": y0, "w": x1-x0, "h": y1-y0}


def overlap(a: dict, b: dict) -> float:
    inter = max(0, min(a["x"]+a["w"], b["x"]+b["w"]) - max(a["x"], b["x"])) * max(
        0, min(a["y"]+a["h"], b["y"]+b["h"]) - max(a["y"], b["y"]))
    return inter / max(1, a["w"]*a["h"] + b["w"]*b["h"] - inter)


def deduplicate(boxes: list[dict], threshold: float = 0.4) -> list[dict]:
    kept = []
    for box in sorted(boxes, key=lambda b: -b["confidence"]):
        if not any(box["type"] == old["type"] and overlap(box, old) >= threshold for old in kept):
            kept.append(box)
    return kept


class LocalDetectors:
    def __init__(self, config: dict):
        inventory = Path(config.get("model_inventory", "reports/t07_models.json"))
        self.inventory = json.loads(inventory.read_text(encoding="utf-8"))
        paths = {}
        for kind in ("face", "plate"):
            spec = self.inventory[kind]
            path = Path(spec["path"])
            if not path.is_file():
                raise FileNotFoundError(f"Missing {kind} weights: {path}. Preload models; offline mode never downloads.")
            if sha256_file(path) != spec["sha256"]:
                raise ValueError(f"{kind} model checksum mismatch: {path}")
            paths[kind] = path
        try:
            import onnxruntime as ort
            from open_image_models import create_detector
        except ImportError as error:
            raise ValueError("Install requirements-t07.txt during preparation, before offline inference") from error
        self.face = cv2.FaceDetectorYN.create(str(paths["face"]), "", (320, 320),
                                             float(config.get("face_confidence", 0.9)), 0.3, 5000)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.plate = create_detector(paths["plate"], backend="yolo_v9", class_labels=["plate"],
                                     conf_thresh=float(config.get("plate_confidence", 0.25)),
                                     providers=["CPUExecutionProvider"], sess_options=options)

    def detect(self, image: np.ndarray) -> list[dict]:
        height, width = image.shape[:2]
        _, faces = self.face.detect(cv2.resize(image, (320, 320)))
        boxes = []
        for face in [] if faces is None else faces:
            x, y, w, h = face[:4]
            boxes.append(clip_box({"x": float(x)*width/320, "y": float(y)*height/320,
                                   "w": float(w)*width/320, "h": float(h)*height/320,
                                   "type": "face", "confidence": float(face[-1]), "source": "yunet"}, width, height))
        for plate in self.plate.predict(image):
            bbox = plate.bounding_box
            boxes.append(clip_box({"x": bbox.x1, "y": bbox.y1, "w": bbox.width, "h": bbox.height,
                                   "type": "plate", "confidence": float(plate.confidence), "source": "yolov9"}, width, height))
        return boxes


def synthetic_proposals(image: np.ndarray) -> tuple[list[dict], int]:
    """Find skin-coloured round heads, require paired eye evidence for face redaction.

    Heads without paired eyes remain ambiguous (review), never declared safe.
    Bright rectangular plate-like patches are proposals, not OCR results.
    All constants are fixed in source before evaluation; only dev was inspected.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    skin = cv2.inRange(hsv, np.array([0, 40, 35], np.uint8), np.array([35, 230, 255], np.uint8))
    contours, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes, ambiguous = [], 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if not (7 <= w <= 70 and 8 <= h <= 85 and 0.55 <= w/h <= 1.6 and area/(w*h) >= 0.45):
            continue
        roi = gray[y:y+h, x:x+w]
        # Interior upper band avoids dark contour edges and mouth.
        ya, yb = max(1, int(h*0.20)), max(2, int(h*0.60))
        xa, xb = max(1, int(w*0.16)), max(2, int(w*0.84))
        band = roi[ya:yb, xa:xb]
        if band.size == 0:
            continue
        cutoff = max(0.0, float(np.median(roi))*0.65)
        left = band[:, :max(1, band.shape[1]//2)] < cutoff
        right = band[:, band.shape[1]//2:] < cutoff
        has_eyes = left.any() and right.any()
        if has_eyes:
            boxes.append(clip_box({"x": x, "y": y, "w": w, "h": h, "type": "face",
                                   "confidence": 0.55, "source": "synthetic_head_eye_rule"}, width, height))
        else:
            ambiguous += 1
    # Low-saturation bright regions; no position-based assumptions about plate location.
    white = cv2.inRange(hsv, np.array([0, 0, 135], np.uint8), np.array([179, 75, 255], np.uint8))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((2, 3), np.uint8))
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (10 <= w <= 110 and 4 <= h <= 28 and 1.7 <= w/h <= 7.5):
            continue
        roi = gray[y:y+h, x:x+w]
        if float(roi.std()) < 18 or np.mean(roi < np.median(roi)*0.65) < 0.03:
            continue
        boxes.append(clip_box({"x": x, "y": y, "w": w, "h": h, "type": "plate",
                               "confidence": 0.5, "source": "synthetic_plate_rule"}, width, height))
    return boxes, ambiguous
