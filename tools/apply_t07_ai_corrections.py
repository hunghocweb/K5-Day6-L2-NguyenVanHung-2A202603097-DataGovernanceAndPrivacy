"""Apply explicit per-image AI correction proposals; preserve the frozen machine run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd

from src.common.contact_sheet import render_contact_sheet
from src.common.io import sha256_file, write_json
from src.common.manifest import verify_manifest
from src.common.review import AuditLog
from src.t07_detectors import clip_box
from src.t07_pipeline import PRED_COLUMNS, blur, digest


def apply(input_dir, machine_dir, output_dir, proposal_path, review_summary_path):
    input_dir, machine_dir, output_dir = map(Path, (input_dir, machine_dir, output_dir))
    if output_dir.resolve() == machine_dir.resolve():
        raise ValueError("AI correction output must be separate from the frozen machine run")
    if verify_manifest(input_dir / "manifest.csv"):
        raise ValueError("Input manifest failed")
    proposal = json.loads(Path(proposal_path).read_text(encoding="utf-8"))
    summary = json.loads(Path(review_summary_path).read_text(encoding="utf-8"))
    machine_path = machine_dir / "predictions_machine.json"
    machine = json.loads(machine_path.read_text(encoding="utf-8"))
    evidence = {item["image_id"]: item for item in summary["evidence"]}
    if set(proposal["corrections"]) != set(evidence):
        raise ValueError("Corrections must match the reviewed image set exactly")
    images = pd.read_csv(input_dir / "images.csv")
    sizes = images.set_index("image_id")
    if set(machine) != set(images.image_id):
        raise ValueError("Machine predictions must cover the same input pool")
    final = {key: [dict(box) for box in boxes] for key, boxes in machine.items()}
    plans = []
    for item_id, values in proposal["corrections"].items():
        size = sizes.loc[item_id]
        observed = evidence[item_id]
        if sha256_file(input_dir / size.path) != observed["original_sha256"]:
            raise ValueError(f"Reviewed source changed: {item_id}")
        if sha256_file(machine_dir / "redacted" / f"{item_id}.png") != observed["exported_sha256"]:
            raise ValueError(f"Reviewed export changed: {item_id}")
        boxes = []
        for x, y, w, h, kind in values:
            if kind not in {"face", "plate"}:
                raise ValueError("Correction type must be face or plate")
            box = {"x": x, "y": y, "w": w, "h": h, "type": kind}
            clipped = clip_box(box, int(size.width), int(size.height))
            if any(box[key] != clipped[key] for key in ("x", "y", "w", "h")):
                raise ValueError(f"Correction out of image bounds: {item_id}")
            boxes.append(box | {"confidence": 1.0, "source": "ai_visual_correction"})
        final[item_id] = boxes
        plans.append({"item_id": item_id, "before": machine[item_id], "after": boxes,
                      "source_image_sha256": observed["original_sha256"],
                      "operation_id": digest({"item_id": item_id, "after": boxes,
                                              "source": observed["original_sha256"],
                                              "machine_sha256": sha256_file(machine_path)})})
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = AuditLog(output_dir / "audit_log.jsonl")
    latest = {event["item_id"]: event for event in audit.read() if event["action"] == "ai_correction:replace"}
    rows = []
    for row in images.itertuples(index=False):
        image = cv2.imread(str(input_dir / row.path))
        if image is None or image.shape[:2] != (row.height, row.width):
            raise ValueError(f"Invalid image: {row.image_id}")
        target = output_dir / "redacted" / f"{row.image_id}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target), blur(image, final[row.image_id])):
            raise ValueError(f"Cannot export {target}")
        rows.extend({"image_id": row.image_id, **box} for box in final[row.image_id])
    pd.DataFrame(rows, columns=PRED_COLUMNS).to_csv(output_dir / "redactions.csv", index=False)
    for plan in plans:
        previous = latest.get(plan["item_id"])
        if previous is None or previous["extra"].get("operation_id") != plan["operation_id"]:
            audit.log(actor=proposal["actor"], action="ai_correction:replace", item_id=plan["item_id"],
                      before=previous["after"] if previous else plan["before"], after=plan["after"],
                      reason="Per-image correction following AI visual rejection notes",
                      operation_id=plan["operation_id"], reviewer_kind="ai", human_review=False,
                      source_image_sha256=plan["source_image_sha256"])
    queue = pd.read_csv(machine_dir / "review_queue.csv", dtype=str).fillna("")
    queue["evidence_path"] = queue.item_id.map(lambda key: str((output_dir / "redacted" / f"{key}.png").resolve()))
    queue.to_csv(output_dir / "correction_queue.csv", index=False)
    render_contact_sheet(queue, output_dir / "contact_sheet.html", before_image_col="original_path",
                         title="T07 AI per-image corrections - original and corrected export", thumb_width=640)
    result = {"n_images": len(images), "n_corrected_images": len(plans), "n_regions": len(rows),
              "actor": proposal["actor"], "reviewer_kind": "ai", "human_review": False,
              "machine_run_preserved": True, "source_machine_sha256": sha256_file(machine_path),
              "proposal_sha256": sha256_file(proposal_path), "quality_metrics": None,
              "formal_release_decision": "REWORK", "post_correction_visual_review": "not_recorded_yet"}
    write_json(output_dir / "correction_manifest.json", result)
    write_json(output_dir / "correction_diff.json", plans)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="fixtures/t07_redaction")
    parser.add_argument("--machine", default="out/poc")
    parser.add_argument("--output", default="out/poc_ai_corrected")
    parser.add_argument("--proposals", default="reports/t07_ai_bbox_corrections.json")
    parser.add_argument("--review-summary", default="out/ai_review/review_summary.json")
    args = parser.parse_args()
    print(json.dumps(apply(args.input, args.machine, args.output, args.proposals, args.review_summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
