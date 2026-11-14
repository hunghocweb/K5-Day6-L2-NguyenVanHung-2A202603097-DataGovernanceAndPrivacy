"""Evaluate frozen machine variants against actual user-approved queue boxes.

This is an internal reviewed-reference experiment, not instructor-GT evaluation.
Never uses machine predictions, rejected proposals or unreviewed images as GT.
"""
from __future__ import annotations

import collections
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd

from src.metrics.detection import over_redaction_rate, pii_miss_rate
from src.t07_detectors import LocalDetectors, clip_box
from src.t07_router import RouterConfig, detect_routed, route_image
from src.t07_rules_v3 import detect as detect_2d


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures/t07_redaction"
OUTPUT = ROOT / "out/t07_reviewed_experiment"
EVIDENCE = ROOT / "reports/evidence/reviewed_reference"
COLS = ["image_id", "x", "y", "w", "h", "type"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def build_reference(images):
    journal = ROOT / "out/web_review/audit_log.jsonl"
    latest = {}
    for line in journal.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("action", "").startswith("human_review:"):
            latest[event["item_id"]] = event
    queue = [r["item_id"] for r in read_rows(ROOT / "out/poc/review_queue.csv")]
    if len(queue) != 15 or len(set(queue)) != 15:
        raise ValueError("Expected the original 15-image review queue")
    references, provenance = [], []
    for key in queue:
        event = latest[key]
        extra = event.get("extra", {})
        if (event["decision"] not in {"approve", "edit"}
                or extra.get("reviewer_kind") != "human" or extra.get("human_review") is not True):
            raise ValueError(f"Queue item has no accepted human review: {key}")
        info = images[key]
        source = FIXTURE / info["path"]
        if event["source_image_sha256"] != sha(source):
            raise ValueError(f"Reviewed image changed: {key}")
        for index, box in enumerate(event["after"]):
            if box["type"] not in {"face", "plate"}:
                raise ValueError(f"Unsupported reference type: {key}")
            clipped = clip_box(box, int(info["width"]), int(info["height"]))
            if any(float(clipped[k]) != float(box[k]) for k in ["x", "y", "w", "h"]):
                raise ValueError(f"Reference box must be valid integer geometry: {key}")
            references.append({"image_id": key, "region_id": f"{key}_review_{index:02d}",
                               **{k: box[k] for k in ["x", "y", "w", "h", "type"]},
                               "must_redact": 1, "split": info["split"]})
        provenance.append({"image_id": key, "split": info["split"], "reviewer": event["actor"],
                           "decision": event["decision"], "reviewed_at": event["ts"],
                           "source_image_sha256": event["source_image_sha256"],
                           "review_snapshot": event["snapshot"], "n_regions": len(event["after"])})
    excluded = [{"image_id": key, "split": images[key]["split"],
                 "reason": ("outside_queue_edit_has_unresolved_note" if latest[key]["decision"] == "edit"
                            else "latest_review_" + latest[key]["decision"])}
                for key in latest if key not in queue]
    return pd.DataFrame(references), provenance, excluded


def frame(predictions, keys):
    return pd.DataFrame([{"image_id": key, **box} for key in keys for box in predictions[key]], columns=COLS)


def score(reference, predictions, keys, sizes):
    gt = reference[reference.image_id.isin(keys)]
    pred = frame(predictions, keys)
    miss = pii_miss_rate(gt, pred, sizes, min_coverage=0.9)
    over = over_redaction_rate(gt, pred, sizes)
    per_image = []
    for key in keys:
        regions = miss["regions"][miss["regions"].image_id == key]
        utility = over_redaction_rate(gt[gt.image_id == key], pred[pred.image_id == key], sizes)
        per_image.append({"image_id": key, "n_gt": len(regions), "missed": int(regions.missed.sum()),
                          "mean_coverage": float(regions.coverage.mean()),
                          "outside_pixels": utility["numerator"], "redacted_pixels": utility["denominator"]})
    # Resample images rather than treating correlated regions as independent.
    rng = np.random.default_rng(42)
    counts = np.array([[r["missed"], r["n_gt"], r["outside_pixels"], r["redacted_pixels"]]
                       for r in per_image], dtype=np.float64)
    boot = counts[rng.integers(0, len(keys), size=(1000, len(keys)))].sum(axis=1)
    miss_ci = np.quantile(boot[:, 0] / boot[:, 1], [0.025, 0.975]).tolist()
    over_ci = np.quantile(np.divide(boot[:, 2], boot[:, 3], out=np.zeros(1000), where=boot[:, 3] != 0),
                          [0.025, 0.975]).tolist()
    slices = {}
    for kind in ["face", "plate"]:
        sub = miss["regions"][miss["regions"].type == kind]
        slices[kind] = {"n_regions": len(sub), "missed": int(sub.missed.sum()),
                        "miss_rate": float(sub.missed.mean()) if len(sub) else None}
    return {"n_images": len(keys), "n_reference_regions": len(gt), "n_predictions": len(pred),
            "missed_regions": miss["numerator"], "pii_miss_rate": float(miss["value"]),
            "region_success_rate": 1.0 - float(miss["value"]),
            "over_redaction_rate": over["value"], "over_outside_pixels": over["numerator"],
            "redacted_pixels": over["denominator"], "guardrail_pass": over["value"] <= 0.30,
            "all_regions_covered_images": sum(r["missed"] == 0 for r in per_image),
            "bootstrap_95ci_miss": miss_ci, "bootstrap_95ci_over": over_ci,
            "by_type": slices, "per_image": per_image}, miss["regions"]


def report(result):
    lines = ["# Thử nghiệm T07 với bbox đã được người dùng duyệt", "",
             "## 1. Nhãn tham chiếu lấy từ đâu?", "",
             "Dùng đúng bbox `after` của quyết định human mới nhất trong queue 15 ảnh: "
             "14 approve và 1 edit. Giữ nguyên split gốc: 9 dev và 6 eval (gọi là val trong yêu cầu). "
             "Không dùng 45 ảnh ngoài queue làm ảnh âm hoặc tự sinh nhãn còn thiếu.", "",
             "Loại pii_00 vì edit có ghi chú chưa xử lý hết; pii_02 và pii_03 đang reject. "
             "Không ghi đè audit/review hoặc GT giảng viên.", "",
             "Đây là **GT nội bộ từ vùng che đã duyệt**, có thể chứa padding và sai sót review. "
             "Nhãn bắt đầu từ dự đoán máy rồi được sửa/duyệt, nên có nguy cơ thiên lệch về hệ thống gốc. "
             "Không xem các số này là điểm chính thức hoặc bằng chứng tổng quát hóa ảnh thật.", "",
             "## 2. Cách đo và khóa cấu hình", "",
             "Seed 42. Miss khi tổng mask che phủ <90% một vùng tham chiếu; success = 1 − miss. "
             "Che thừa = pixel che ngoài hợp vùng tham chiếu / tổng pixel che, guardrail ≤30%. "
             "Dùng hàm metric của repo; không thay bằng IoU hoặc confidence. CI bootstrap theo ảnh, "
             "1000 lần, seed 42. Không có nhãn condition nên không suy đoán slice condition.", "",
             "Nguồn v3/router và tham số được hash trước khi đo dev; chạy dev trước, sau đó eval "
             "với cùng cấu hình, không tune theo kết quả. Router vẫn dùng ngưỡng thử 0,60 ±0,08; "
             "chưa hiệu chỉnh thật/2D vì không có ảnh thật. v3 padding 1% ở cả hai nhánh so sánh router.", "",
             "Baseline và PoC v1 là dự đoán máy đã lưu, không áp sửa human. Hai cấu hình v3 được "
             "chạy mới từ ảnh nguồn. Không chấm bản review15 với chính nó rồi gọi là chất lượng máy.", "",
             "## 3. Kết quả", "",
             "| Tập | Cấu hình | Ảnh / vùng nhãn | Vùng phủ ≥90% | Bỏ sót | Che thừa | Guardrail |",
             "|---|---|---:|---:|---:|---:|---|"]
    for split in ["dev", "eval"]:
        for variant, value in result["metrics"][split].items():
            lines.append(f"| {split} | {variant} | {value['n_images']} / {value['n_reference_regions']} | "
                         f"{100*value['region_success_rate']:.2f}% | {value['missed_regions']}/"
                         f"{value['n_reference_regions']} ({100*value['pii_miss_rate']:.2f}%) | "
                         f"{100*value['over_redaction_rate']:.2f}% | {'PASS' if value['guardrail_pass'] else 'FAIL'} |")
    lines += ["", "### Theo mặt / biển số", "",
              "| Tập | Cấu hình | Loại | Vùng bỏ sót / nhãn | Miss rate |", "|---|---|---|---:|---:|"]
    for split in ["dev", "eval"]:
        for variant, value in result["metrics"][split].items():
            for kind, sub in value["by_type"].items():
                lines.append(f"| {split} | {variant} | {kind} | {sub['missed']}/{sub['n_regions']} | {100*sub['miss_rate']:.2f}% |")
    lines += ["", "### Router", "", f"Route trên toàn bộ 60 ảnh: `{json.dumps(result['routes_all'], ensure_ascii=False)}`.", "",
              "Route là dự đoán từ pixel. Tất cả fixture được mô tả là synthetic trong tài liệu, nhưng "
              "không sử dụng trường này để định tuyến. Nếu router và v3 cho cùng kết quả trên tập này, "
              "chưa có bằng chứng router tăng chất lượng; vẫn cần ảnh thật/ảnh lai để kiểm tra.", "",
              "## 4. Giới hạn xác thực", "",
              "Chỉ 9/36 ảnh dev và 6/24 ảnh eval có nhãn được duyệt trong thử nghiệm. Queue chọn ảnh "
              "rủi ro cao, không phải mẫu ngẫu nhiên đại diện. Không báo số này như kết quả toàn tập.", "",
              "Ảnh trong cả hai nhóm đã được xem/review trước khi xây v3; đặc biệt pii_03 eval từng "
              "được dùng để phân tích lỗi, và các bản sửa/ảnh review khác cũng có thể đã ảnh hưởng "
              "thiết kế rule. Không chứng nhận 6 ảnh eval là holdout độc lập. Lần chạy này chỉ giữ "
              "nguyên tham số giữa dev và eval; không xóa việc đã tiếp xúc dữ liệu trước đó.", "",
              "Để xác thực độc lập cần nhãn giảng viên hoặc tập ảnh mới chưa dùng chỉnh thuật toán. "
              "Không nhận bonus ablation hoặc kết luận hệ thống an toàn chỉ từ thử nghiệm này.", "",
              "## 5. File bằng chứng và tái lập", "",
              "- `reports/t07_reviewed_experiment.json`: số đếm, CI, per-image, timing và provenance.",
              "- `reports/evidence/reviewed_reference/regions.csv`: bbox tham chiếu nội bộ.",
              "- `reports/evidence/reviewed_reference/provenance.json`: quyết định gốc và hash audit.",
              "- `reports/evidence/reviewed_reference/experiment_freeze.json`: hash nguồn/tham số trước dev.",
              "- `out/t07_reviewed_experiment/`: dự đoán, coverage mỗi vùng, route/audit của lần chạy.",
              "", "```powershell", ".\\.venv\\Scripts\\python.exe -m tools.evaluate_t07_reviewed", "```", ""]
    (ROOT / "reports/t07_reviewed_experiment.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    images = {r["image_id"]: r for r in read_rows(FIXTURE / "images.csv")}
    reference, provenance, excluded = build_reference(images)
    reference.to_csv(EVIDENCE / "regions.csv", index=False)
    write_json(EVIDENCE / "provenance.json", {"reference_kind": "user_reviewed_redaction_boxes",
        "journal": "out/web_review/audit_log.jsonl", "journal_sha256": sha(ROOT / "out/web_review/audit_log.jsonl"),
        "included": provenance, "excluded_reviewed_images": excluded,
        "scope": "original top15 risk queue only; no additional review performed",
        "not_official_gt": True, "not_independent_holdout": True})
    config = RouterConfig()
    params = {"face_confidence": 0.90, "plate_confidence": 0.25,
              "model_inventory": "reports/t07_models.json"}
    files = ["src/t07_router.py", "src/t07_rules_v3.py", "src/t07_detectors.py",
             "src/metrics/detection.py", "tools/evaluate_t07_reviewed.py", "reports/t07_models.json"]
    frozen = {"frozen_at": datetime.now(timezone.utc).isoformat(), "seed": 42,
              "router": asdict(config), "v3_padding": 0.01, "min_coverage": 0.90,
              "model_params": params, "source_sha256": {name: sha(ROOT / name) for name in files},
              "reference_sha256": sha(EVIDENCE / "regions.csv"),
              "input_sha256": {key: sha(FIXTURE / row["path"]) for key, row in images.items()},
              "note": "Frozen before this dev/eval measurement; prior exposure remains disclosed, no tuning in this run."}
    write_json(EVIDENCE / "experiment_freeze.json", frozen)
    predictions = {name: json.loads((ROOT / f"out/{folder}/predictions_machine.json").read_text(encoding="utf-8"))
                   for name, folder in [("baseline", "baseline"), ("poc_v1", "poc")]}
    predictions.update(poc_v3={}, poc_v3_router={})
    saved_hashes = {name: sha(ROOT / f"out/{folder}/predictions_machine.json")
                    for name, folder in [("baseline", "baseline"), ("poc_v1", "poc")]}
    sizes = {key: (int(row["height"]), int(row["width"])) for key, row in images.items()}
    detector = None

    def real_detector(image):
        nonlocal detector
        if detector is None:
            detector = LocalDetectors(params)
        return detector.detect(image)

    metrics, routes, timing = {}, {}, {}
    journal_before = sha(ROOT / "out/web_review/audit_log.jsonl")
    for split in ["dev", "eval"]:
        timing[split] = collections.defaultdict(float)
        for key, row in images.items():
            if row["split"] != split:
                continue
            image = cv2.imread(str(FIXTURE / row["path"]))
            if image is None or image.shape[:2] != sizes[key]:
                raise ValueError(f"Invalid image: {key}")
            start = time.perf_counter()
            boxes, _ = detect_2d(image.copy())
            predictions["poc_v3"][key] = [clip_box(b, image.shape[1], image.shape[0], 0.01) for b in boxes]
            timing[split]["poc_v3_seconds"] += time.perf_counter() - start
            start = time.perf_counter()
            result = detect_routed(image, real_detector=real_detector, config=config)
            predictions["poc_v3_router"][key] = [clip_box(b, image.shape[1], image.shape[0], 0.01) for b in result["boxes"]]
            timing[split]["poc_v3_router_seconds"] += time.perf_counter() - start
            routes[key] = {"image_id": key, "split": split, **result["routing"]}
        keys = [r["image_id"] for r in provenance if r["split"] == split]
        metrics[split] = {}
        for variant, pred in predictions.items():
            value, coverage = score(reference, pred, keys, sizes)
            metrics[split][variant] = value
            coverage.to_csv(OUTPUT / f"coverage_{variant}_{split}.csv", index=False)
        write_json(OUTPUT / f"metrics_{split}.json", metrics[split])
        print(json.dumps({"split": split, "reviewed_images": len(keys),
              "results": {k: {m: v[m] for m in ["pii_miss_rate", "over_redaction_rate", "missed_regions", "n_reference_regions"]}
                          for k, v in metrics[split].items()}}, ensure_ascii=True), flush=True)
    if journal_before != sha(ROOT / "out/web_review/audit_log.jsonl"):
        raise ValueError("Human review journal changed during experiment; rerun from a stable snapshot")
    if any(sha(ROOT / path) != digest for path, digest in frozen["source_sha256"].items()):
        raise ValueError("Source changed during measurement")
    for variant, pred in predictions.items():
        frame(pred, list(images)).to_csv(OUTPUT / f"predictions_{variant}.csv", index=False)
    write_json(OUTPUT / "routes.json", routes)
    (OUTPUT / "audit_log.jsonl").write_text("".join(json.dumps({"actor": "machine:t07_router",
        "action": "route_image", "observed_at": datetime.now(timezone.utc).isoformat(),
        "source_image_sha256": frozen["input_sha256"][key], **value}, ensure_ascii=False) + "\n"
        for key, value in routes.items()), encoding="utf-8")
    result = {"created_at": datetime.now(timezone.utc).isoformat(), "experiment": "internal_user_reviewed_reference",
              "seed": 42, "official_gt": False, "independent_holdout": False, "metrics": metrics,
              "routes_all": dict(collections.Counter(r["route"] for r in routes.values())),
              "routes_by_split": {s: dict(collections.Counter(r["route"] for r in routes.values() if r["split"] == s))
                                  for s in ["dev", "eval"]},
              "route_consistency_with_documented_fixture_2d": sum(r["route"] == "2d" for r in routes.values()) / len(routes),
              "real_photo_accuracy": None, "timing_all_images_by_split": timing,
              "saved_prediction_sha256": saved_hashes, "reference_images": provenance,
              "freeze": "reports/evidence/reviewed_reference/experiment_freeze.json",
              "review_journal_unchanged": True,
              "warning": "Risk-selected approved blur boxes, with padding and prior model/review exposure; not independent instructor ground truth."}
    write_json(ROOT / "reports/t07_reviewed_experiment.json", result)
    report(result)
    print("Report: reports/t07_reviewed_experiment.md", flush=True)


if __name__ == "__main__":
    main()
