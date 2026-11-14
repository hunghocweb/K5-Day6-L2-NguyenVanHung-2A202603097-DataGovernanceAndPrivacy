"""Experimental, pixel-only routing between pretrained and flat-2D detectors.

Standalone opt-in module: importing it does not change the existing pipeline.
Default thresholds are provisional, NOT measured routing accuracy. Freeze a
configuration using labelled dev photographs AND drawings before evaluation.
Colour count and Laplacian variance are diagnostics, not universal art tests.

Example integration (caller owns padding, blurring, persistence and review):
    result = detect_routed(image, real_detector=detector.detect)
    boxes = result["boxes"]
    # Persist result["routing"] in both per-image output and audit event.

The selected route is real / 2d / both. No filenames, manifests, splits, review
decisions or ground truth are accepted by route_image(). A missing detection
does not mean an image is safe; the caller must retain its existing risk gate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable

import cv2
import numpy as np

from .t07_detectors import clip_box, overlap


ROUTER_VERSION = "flat-pixels-v1-experimental"


@dataclass(frozen=True)
class RouterConfig:
    flat_threshold: float = 0.60
    uncertainty_margin: float = 0.08
    flat_std: float = 2.0
    patch_size: int = 5
    black_level: int = 8
    border_dark_fraction: float = 0.995
    max_trim_fraction: float = 0.45
    max_feature_side: int = 1024
    # Reference to an external dev-calibration record, empty until measured.
    calibration_reference: str = ""

    def __post_init__(self):
        if not math.isfinite(self.flat_threshold) or not 0 < self.flat_threshold < 1:
            raise ValueError("flat_threshold must be between 0 and 1")
        if (not math.isfinite(self.uncertainty_margin)
                or not 0 < self.uncertainty_margin < min(
                    self.flat_threshold, 1 - self.flat_threshold)):
            raise ValueError("uncertainty_margin must fit around flat_threshold")
        if not math.isfinite(self.flat_std) or self.flat_std <= 0:
            raise ValueError("flat_std must be positive")
        if (not isinstance(self.patch_size, int) or self.patch_size < 3
                or self.patch_size % 2 != 1):
            raise ValueError("patch_size must be an odd integer >= 3")
        if not 0 <= self.black_level <= 255:
            raise ValueError("black_level must be in [0, 255]")
        if not 0.9 <= self.border_dark_fraction <= 1:
            raise ValueError("border_dark_fraction must be in [0.9, 1]")
        if not 0 <= self.max_trim_fraction < 0.5:
            raise ValueError("max_trim_fraction must be in [0, 0.5)")
        if self.max_feature_side < 32:
            raise ValueError("max_feature_side must be >= 32")


def _validate_image(image: np.ndarray) -> None:
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3
            or min(image.shape[:2]) < 1):
        raise ValueError("Expected a nonempty uint8 BGR image with three channels")


def _content_roi(image: np.ndarray, config: RouterConfig):
    """Trim consecutive almost-black outer rows/columns, retaining dark content.

    Trimming is capped on each side. Only feature computation uses this crop;
    both detectors still receive the complete original image and coordinates.
    """
    height, width = image.shape[:2]
    dark = np.max(image, axis=2) <= config.black_level
    top, bottom, left, right = 0, height, 0, width
    max_y, max_x = int(height * config.max_trim_fraction), int(width * config.max_trim_fraction)
    while top < max_y and float(dark[top].mean()) >= config.border_dark_fraction:
        top += 1
    while height - bottom < max_y and float(dark[bottom - 1].mean()) >= config.border_dark_fraction:
        bottom -= 1
    while left < max_x and float(dark[top:bottom, left].mean()) >= config.border_dark_fraction:
        left += 1
    while width - right < max_x and float(dark[top:bottom, right - 1].mean()) >= config.border_dark_fraction:
        right -= 1
    return image[top:bottom, left:right], [left, top, right - left, bottom - top]


def route_image(image_bgr: np.ndarray, config: RouterConfig | None = None) -> dict:
    """Return a JSON-serializable routing decision with measured pixel features.

    flat_fraction is a heuristic score, NOT a probability. Near-threshold or
    low-information inputs run both branches. Quantised colours and Laplacian
    are logged for dev analysis; their thresholds have not been calibrated, so
    this first version deliberately does not invent additional cut-offs.
    """
    _validate_image(image_bgr)
    config = config or RouterConfig()
    roi, crop = _content_roi(image_bgr, config)
    scale = min(1.0, config.max_feature_side / max(roi.shape[:2]))
    if scale < 1:
        roi = cv2.resize(roi, (max(1, round(roi.shape[1] * scale)),
                               max(1, round(roi.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    kernel = (config.patch_size, config.patch_size)
    mean = cv2.blur(gray, kernel, borderType=cv2.BORDER_REFLECT_101)
    square_mean = cv2.blur(gray * gray, kernel, borderType=cv2.BORDER_REFLECT_101)
    local_std = np.sqrt(np.maximum(square_mean - mean * mean, 0))
    flat_fraction = float(np.mean(local_std < config.flat_std))
    quantized = roi.astype(np.uint16) // 8
    codes = quantized[:, :, 0] + 32 * quantized[:, :, 1] + 1024 * quantized[:, :, 2]
    unique_colors = int(np.unique(codes).size)
    global_std = float(gray.std())
    features = {
        "flat_fraction": flat_fraction,
        "quantized_unique_colors": unique_colors,
        "quantized_color_occupancy": unique_colors / min(gray.size, 32768),
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
        "gray_std": global_std,
        "content_roi_xywh": crop,
        "feature_shape_hw": list(gray.shape),
        "feature_scale": scale,
    }
    if min(gray.shape) < 2 * config.patch_size or global_std < config.flat_std:
        route, reason = "both", "low_information"
    elif abs(flat_fraction - config.flat_threshold) <= config.uncertainty_margin:
        route, reason = "both", "near_flat_threshold"
    elif flat_fraction > config.flat_threshold:
        route, reason = "2d", "high_flat_fraction"
    else:
        route, reason = "real", "low_flat_fraction"
    return {
        "route": route, "reason": reason, "router_version": ROUTER_VERSION,
        "features": features, "config": asdict(config),
        "calibration_status": ("reference_supplied_not_verified"
                               if config.calibration_reference else "uncalibrated"),
    }


def detect_2d(image_bgr: np.ndarray) -> tuple[list[dict], int]:
    """Use existing v3 ellipse/facial-mark and white-rectangle/text algorithms."""
    from .t07_rules_v3 import detect

    return detect(image_bgr)


def _coverage_preserving_nms(boxes: list[dict], width: int, height: int,
                             threshold: float) -> list[dict]:
    """Merge same-class overlaps using their union rather than discard a tail.

    This conservative variant can over-redact; distinct overlapping identities
    are not recovered here. Unlike hard NMS, it never removes covered pixels.
    """
    remaining = sorted((clip_box(dict(b), width, height) for b in boxes),
                       key=lambda b: (-float(b["confidence"]), str(b["type"]),
                                      b["x"], b["y"], b["w"], b["h"]))
    kept = []
    while remaining:
        current = remaining.pop(0)
        sources = set(current.get("merged_sources", [current.get("source", "unknown")]))
        changed = True
        while changed:
            changed = False
            for index, other in enumerate(remaining):
                if current["type"] != other["type"] or overlap(current, other) < threshold:
                    continue
                x, y = min(current["x"], other["x"]), min(current["y"], other["y"])
                right = max(current["x"] + current["w"], other["x"] + other["w"])
                bottom = max(current["y"] + current["h"], other["y"] + other["h"])
                sources.update(other.get("merged_sources", [other.get("source", "unknown")]))
                current.update(x=x, y=y, w=right - x, h=bottom - y,
                               confidence=max(float(current["confidence"]), float(other["confidence"])),
                               source="router_union")
                remaining.pop(index)
                changed = True
                break
        current["merged_sources"] = sorted(sources)
        kept.append(current)
    return kept


def detect_routed(image_bgr: np.ndarray, *,
                  real_detector: Callable[[np.ndarray], list[dict]],
                  config: RouterConfig | None = None,
                  synthetic_detector: Callable[[np.ndarray], tuple[list[dict], int]] | None = None,
                  merge_iou: float = 0.4) -> dict:
    """Dispatch lazily, returning detections and route metadata without any I/O.

    real_detector should be the existing YuNet+YOLO callback. No weights are
    loaded/downloaded here. Caller must write routing metadata to its real
    output/audit, and enforce review; returning metadata does not persist it.
    """
    if not math.isfinite(merge_iou) or not 0 < merge_iou <= 1:
        raise ValueError("merge_iou must be in (0, 1]")
    routing = route_image(image_bgr, config)
    route = routing["route"]
    boxes, ambiguous = [], 0
    if route in ("real", "both"):
        boxes.extend(real_detector(image_bgr.copy()))
    if route in ("2d", "both"):
        synthetic, ambiguous = (synthetic_detector or detect_2d)(image_bgr.copy())
        boxes.extend(synthetic)
    height, width = image_bgr.shape[:2]
    if route == "both":
        boxes = _coverage_preserving_nms(boxes, width, height, merge_iou)
    else:
        boxes = [clip_box(dict(box), width, height) for box in boxes]
    return {"boxes": boxes, "ambiguous_heads": int(ambiguous),
            "routing": routing, "requires_review": route == "both" or ambiguous > 0 or not boxes}
