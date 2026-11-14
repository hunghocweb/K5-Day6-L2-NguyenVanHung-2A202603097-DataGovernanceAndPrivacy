"""Offline development-only model probe; detection counts are NOT quality metrics."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import pandas as pd
import onnxruntime as ort
from open_image_models import create_detector

from src.common.io import sha256_file
from src.common.offline import enable_offline_guard


def main():
    enable_offline_guard()
    weights = Path("models_optional")
    face_path = weights / "face_detection_yunet_2023mar.onnx"
    plate_path = weights / "yolo-v9-t-640-license-plates-end2end.onnx"
    for path in (face_path, plate_path):
        if not path.is_file():
            raise FileNotFoundError(f"Preload weights before running offline: {path}")
    face = cv2.FaceDetectorYN.create(str(face_path), "", (320, 320), 0.9, 0.3, 5000)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    plate = create_detector(plate_path, backend="yolo_v9", class_labels=["plate"],
                            conf_thresh=0.25, providers=["CPUExecutionProvider"], sess_options=options)
    table = pd.read_csv("fixtures/t07_redaction/images.csv")
    dev = table[table.split == "dev"].sort_values("image_id")
    rows = []
    started = time.perf_counter()
    for row in dev.itertuples(index=False):
        image = cv2.imread(str(Path("fixtures/t07_redaction") / row.path))
        _, faces = face.detect(cv2.resize(image, (320, 320)))
        plates = plate.predict(image)
        rows.append({"image_id": row.image_id, "split": "dev",
                     "n_faces": 0 if faces is None else len(faces), "n_plates": len(plates)})
    out = Path("out/t07_inspection")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "pretrained_dev_probe.csv", index=False)
    result = {"n_dev": len(rows), "face_detections": sum(r["n_faces"] for r in rows),
              "plate_detections": sum(r["n_plates"] for r in rows),
              "runtime_seconds": time.perf_counter() - started,
              "models": {p.name: sha256_file(p) for p in (face_path, plate_path)},
              "opencv": cv2.__version__, "onnxruntime": ort.__version__,
              "quality_metrics": None, "ground_truth_read": False, "network": "blocked"}
    (out / "pretrained_dev_probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
