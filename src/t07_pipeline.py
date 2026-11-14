"""T07 load -> detect -> rank -> human overrides -> export and risk gate."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd

from .common.contact_sheet import render_contact_sheet
from .common.decision import Decision, write_decision
from .common.io import sha256_file, write_json
from .common.manifest import verify_manifest
from .common.review import load_review_decisions, make_review_queue, write_review_queue
from .t07_detectors import LocalDetectors, clip_box, deduplicate, synthetic_proposals

PRED_COLUMNS = ["image_id", "x", "y", "w", "h", "type", "confidence", "source"]


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def load(ctx):
    problems = verify_manifest(ctx.input_dir / "manifest.csv")
    if problems:
        raise ValueError(f"T07 provenance failed: {problems[:3]}")
    images = pd.read_csv(ctx.input_dir / "images.csv")
    required = {"image_id", "path", "width", "height", "split"}
    if not required.issubset(images.columns) or images.image_id.duplicated().any():
        raise ValueError("Invalid T07 images.csv schema or duplicate image_id")
    if not set(images.split).issubset({"dev", "eval"}):
        raise ValueError("T07 split must be dev or eval")
    selected = ctx.param("processing_split", "all")
    if selected not in {"all", "dev", "eval"}:
        raise ValueError("processing_split must be all/dev/eval")
    if selected != "all":
        images = images[images.split == selected]
    if images.empty:
        raise ValueError("No input images in selected split")
    root = ctx.input_dir.resolve()
    for row in images.itertuples(index=False):
        path = (root / row.path).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Image path outside input directory")
    return images.sort_values("image_id").reset_index(drop=True)


def blur(image, boxes):
    result = image.copy()
    for box in boxes:
        x, y, w, h = (int(box[k]) for k in ("x", "y", "w", "h"))
        result[y:y+h, x:x+w] = cv2.GaussianBlur(result[y:y+h, x:x+w], (0, 0), sigmaX=12, sigmaY=12)
    return result


def process(ctx, images, mode="poc"):
    params = ctx.config.get("params", {})
    precision_mode = mode == "poc" and params.get("synthetic_algorithm") == "precision"
    auto_mode = mode == "poc" and params.get("detection_pipeline") == "auto"
    auto_detector = None
    if auto_mode:
        from .t07_auto import AutoDetectors
        auto_detector = AutoDetectors(params)
    detector_class = LocalDetectors
    if params.get("detector_strategy") == "multiscale":
        from .t07_multiscale import MultiScaleDetectors
        detector_class = MultiScaleDetectors
    detector = None if mode == "blur_all" or precision_mode or auto_mode else detector_class(params)
    model_hashes = {} if detector is None else {
        kind: detector.inventory[kind]["sha256"] for kind in ("face", "plate")}
    if precision_mode:
        model_hashes = {"precision_config": sha256_file(Path(__file__).resolve().parents[1] / "configs/t07_precision.json")}
    rows, boxes_by_id = [], {}
    precision_trace = {}
    for row in images.itertuples(index=False):
        path = ctx.input_dir / row.path
        image = cv2.imread(str(path))
        if image is None or image.shape[:2] != (int(row.height), int(row.width)):
            raise ValueError(f"Unreadable image or incorrect dimensions: {path}")
        ambiguous = 0
        if mode == "blur_all":
            boxes = [{"x": 0, "y": 0, "w": row.width, "h": row.height,
                      "type": "all", "confidence": 1.0, "source": "blur_all_control"}]
        elif auto_mode:
            result = auto_detector.detect_details(image)
            boxes = [clip_box(dict(b), row.width, row.height, float(params.get("padding", 0))) for b in result["merged"]]
            ambiguous = result["ambiguous_heads"]
            precision_trace[row.image_id] = {"algorithm": result["algorithm"], "routing": result["routing"],
                "cascade": result.get("cascade"), "selected_algorithm": result.get("selected_algorithm"),
                "detector_padding_applied": result.get("detector_padding_applied", False),
                "padding_policy": result.get("padding_policy"),
                "review_reasons": result["review_reasons"], "requires_human_review": True,
                "human_review_recorded": False, "bbox_calibration_applied": False,
                "padding_applications": None if result.get("detector_padding_applied") else 1,
                "user_padding_applications": 1}
            if auto_detector.photo is not None:
                model_hashes = {kind: auto_detector.photo.inventory[kind]["sha256"] for kind in ("face", "plate")}
        elif precision_mode:
            from .t07_precision_pipeline import detect_details
            result = detect_details(image)
            boxes, ambiguous = result["boxes"], result["ambiguous_heads"]
            boxes = [clip_box(b, row.width, row.height, float(params.get("padding", 0))) for b in boxes]
            precision_trace[row.image_id] = {"algorithm": result["algorithm"], "route": "2d",
                "route_source": "explicit_precision_2d_mode", "estimated_sigma": result["estimated_sigma"],
                "requires_human_review": True, "human_review_recorded": False}
        else:
            boxes = detector.detect(image)
            if mode == "poc":
                if params.get("synthetic_algorithm", "v1") == "v3":
                    from .t07_rules_v3 import detect
                    boxes, ambiguous = detect(image, boxes)
                elif params.get("synthetic_algorithm", "v1") == "v2":
                    from .t07_rules_v2 import detect
                    boxes, ambiguous = detect(image, boxes)
                elif params.get("synthetic_algorithm", "v1") == "v1":
                    proposals, ambiguous = synthetic_proposals(image)
                    boxes = deduplicate(boxes + proposals)
            padding = float(params.get("padding", 0.03)) if mode == "poc" else 0.0
            boxes = [clip_box(box, row.width, row.height, padding) for box in boxes]
        boxes_by_id[row.image_id] = boxes
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dark = float(gray.mean()) < 65
        small = sum(min(box["w"], box["h"]) < 16 for box in boxes)
        low = sum(box["confidence"] < 0.7 for box in boxes)
        no_face = not any(box["type"] == "face" for box in boxes)
        # Frozen input-only risk signals; no GT or eval performance is consulted.
        risk = 3.0*int(no_face) + 2.0*ambiguous + 1.5*int(dark) + 0.5*small + 0.25*low
        fingerprint = digest({"image_sha256": sha256_file(path), "models": model_hashes,
                              "params": params, "mode": mode, "boxes": boxes})
        reason = f"no_face={int(no_face)};ambiguous_heads={ambiguous};dark={int(dark)};small={small};snapshot={fingerprint}"
        after = ctx.out(f"machine_images/{row.image_id}.png")
        if not cv2.imwrite(str(after), blur(image, boxes)):
            raise ValueError(f"Cannot write {after}")
        rows.append({"image_id": row.image_id, "score": risk, "reason": reason,
                     "split": row.split, "evidence_path": str(after.resolve()),
                     "original_path": str(path.resolve()), "snapshot": fingerprint})
    ctx.stats["n_items"] = len(images)
    ctx.stats["mode"] = mode
    ctx.stats["model_hashes"] = model_hashes
    if precision_mode:
        ctx.stats["algorithm"] = "precision-dev-1"
        write_json(ctx.out("precision_trace.json"), precision_trace)
    if auto_mode:
        from .t07_auto import VERSION
        ctx.stats["algorithm"] = VERSION
        write_json(ctx.out("routing_trace.json"), precision_trace)
    write_json(ctx.out("predictions_machine.json"), boxes_by_id)
    return pd.DataFrame(rows), boxes_by_id


def rank(ctx, scored):
    queue = make_review_queue(scored, "image_id", "score", int(ctx.param("budget", 15)),
                              reason_col="reason", slice_col="split", evidence_col="evidence_path")
    queue["original_path"] = queue.item_id.map(scored.set_index("image_id").original_path)
    return queue


def render_review(ctx, queue):
    """Publish the queue only after final images exist, keeping raw scores/snapshots."""
    current = queue.copy()
    current["evidence_path"] = current.item_id.map(lambda item: str(ctx.out(f"redacted/{item}.png").resolve()))
    write_review_queue(current, ctx.out("review_queue.csv"))
    render_contact_sheet(current, ctx.out("contact_sheet.html"), before_image_col="original_path",
                         title="T07 review - original and current exported image", thumb_width=640)


def reviewed_boxes(value, width, height):
    try:
        boxes = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("override_value must be a JSON list of ALL final boxes for the image") from error
    if not isinstance(boxes, list):
        raise ValueError("override_value must be a list; use [] to remove all boxes")
    out = []
    for box in boxes:
        if not isinstance(box, dict) or not {"x", "y", "w", "h", "type"}.issubset(box):
            raise ValueError("Each review box needs x,y,w,h,type")
        if box["type"] not in {"face", "plate"}:
            raise ValueError("Review type must be face or plate")
        checked = clip_box(box, width, height)
        # Reject out-of-bounds values instead of silently changing a human edit.
        if any(float(box[k]) != checked[k] for k in ("x", "y", "w", "h")):
            raise ValueError("Human boxes must be integer pixels fully inside the image")
        out.append({k: checked[k] for k in ("x", "y", "w", "h", "type")} |
                   {"confidence": 1.0, "source": "human"})
    return out


def review(ctx, machine, queue, images):
    final = {key: [dict(box) for box in value] for key, value in machine.items()}
    path = ctx.output_dir / "review_queue_reviewed.csv"
    reviewed = load_review_decisions(path) if path.exists() else pd.DataFrame()
    completed = set()
    history = ctx.audit.read()
    existing_operations = {event.get("extra", {}).get("operation_id") for event in history}
    latest = {}
    for event in history:
        if event.get("action", "").startswith("review:"):
            latest[event["item_id"]] = event
    expected = queue.set_index("item_id")
    sizes = images.set_index("image_id")
    if not reviewed.empty and reviewed.item_id.duplicated().any():
        raise ValueError("Duplicate image_id in reviewed queue")
    operations, edits, requests = [], [], []
    for row in reviewed.itertuples(index=False):
        if row.item_id not in expected.index or row.reason != expected.loc[row.item_id, "reason"]:
            raise ValueError(f"Unknown or stale reviewed item: {row.item_id}; review current queue")
        if not row.reviewed_at:
            raise ValueError("reviewed_at is required for every human decision")
        try:
            timestamp = datetime.fromisoformat(row.reviewed_at.replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                raise ValueError("Review time needs a timezone")
        except ValueError as error:
            raise ValueError("reviewed_at must be an ISO timestamp with timezone") from error
        if row.reviewer_decision in {"edit", "reject"}:
            if not row.review_note.strip():
                raise ValueError("A review note is required when rejecting or changing redactions")
            if row.reviewer_decision == "edit" or row.override_value.strip():
                size = sizes.loc[row.item_id]
                final[row.item_id] = reviewed_boxes(row.override_value, int(size.width), int(size.height))
        elif row.override_value.strip():
            raise ValueError("Overrides require edit/reject; approve/skip/escalate cannot change boxes")
        if not row.override_value.strip() and row.reviewer_decision != "edit":
            previous = latest.get(row.item_id)
            if previous and previous.get("extra", {}).get("machine_snapshot") == row.reason:
                # The refreshed page shows the last exported correction, not the raw model image.
                final[row.item_id] = [dict(box) for box in previous["after"]]
        if row.reviewer_decision in {"approve", "edit"} or (row.reviewer_decision == "reject" and row.override_value.strip()):
            completed.add(row.item_id)
        payload = {"item_id": row.item_id, "reviewer": row.reviewer.strip(),
                   "decision": row.reviewer_decision, "reviewed_at": row.reviewed_at,
                   "note": row.review_note, "snapshot": row.reason,
                   "before": machine[row.item_id], "after": final[row.item_id]}
        operation_id = digest(payload)
        operations.append((operation_id, payload))
        if row.reviewer_decision == "reject" and not row.override_value.strip():
            requests.append({"item_id": row.item_id, "reviewer": row.reviewer.strip(),
                             "reviewed_at": row.reviewed_at, "review_note": row.review_note,
                             "operation_id": operation_id, "status": "needs_correction_then_rereview"})
        if final[row.item_id] != machine[row.item_id]:
            edits.append(payload)
    # Validate the entire review file before appending audit events.
    for operation_id, payload in operations:
        previous = latest.get(payload["item_id"])
        previous_id = previous.get("extra", {}).get("operation_id") if previous else None
        if previous is None or previous_id != operation_id or previous["after"] != payload["after"]:
            ctx.audit.log(actor=payload["reviewer"], action=f"review:{payload['decision']}",
                          item_id=payload["item_id"], before=previous["after"] if previous else payload["before"],
                          after=payload["after"], reason=payload["note"], reviewed_at=payload["reviewed_at"],
                          operation_id=operation_id, machine_snapshot=payload["snapshot"],
                          reapplied=operation_id in existing_operations)
    reviewed_ids = set(reviewed.item_id) if not reviewed.empty else set()
    for item_id, boxes in final.items():
        previous = latest.get(item_id)
        if item_id not in reviewed_ids and previous and previous["after"] != boxes:
            # Removing a reviewed row/file also changes the export; record the reset as a machine action.
            ctx.audit.log(actor="pipeline", action="review:reset", item_id=item_id,
                          before=previous["after"], after=boxes, reason="No current review decision; restored machine output",
                          operation_id=digest({"reset": item_id, "after": boxes}),
                          machine_snapshot=str(expected.loc[item_id, "reason"]) if item_id in expected.index else "")
    state = {"queue_size": len(queue), "n_completed": len(completed),
             "n_pending": len(queue)-len(completed), "n_changed_images": len(edits),
             "n_correction_requests": len(requests),
             "independent_quality_validation": False,
             "quality_claim": "Human queue completion is not a full-dataset privacy guarantee"}
    write_json(ctx.out("review_state.json"), state)
    write_json(ctx.out("review_diff.json"), edits)
    pd.DataFrame(requests, columns=["item_id", "reviewer", "reviewed_at", "review_note", "operation_id", "status"]).to_csv(
        ctx.out("human_change_requests.csv"), index=False)
    ctx.stats.update(state)
    return final


def export(ctx, images, boxes_by_id):
    rows = []
    for row in images.itertuples(index=False):
        image = cv2.imread(str(ctx.input_dir / row.path))
        boxes = boxes_by_id[row.image_id]
        after = ctx.out(f"redacted/{row.image_id}.png")
        if not cv2.imwrite(str(after), blur(image, boxes)):
            raise ValueError(f"Cannot write {after}")
        rows.extend({"image_id": row.image_id, **box} for box in boxes)
    pred = pd.DataFrame(rows, columns=PRED_COLUMNS)
    pred.to_csv(ctx.out("redactions.csv"), index=False)
    counts = pred.groupby(["type", "source"]).size().reset_index(name="n_regions").to_dict("records") if len(pred) else []
    write_json(ctx.out("detection_summary.json"), {"n_images": len(images), "n_regions": len(pred),
               "split_counts": images.split.value_counts().to_dict(), "detections_by_source": counts,
               "quality_metrics": None, "ground_truth_read": False})
    ctx.stats["n_regions"] = len(pred)


def decide(ctx):
    pending = ctx.stats.get("n_pending", 0)
    requests = ctx.stats.get("n_correction_requests", 0)
    code = "HUMAN_CORRECTIONS_REQUIRED" if requests else ("HUMAN_REVIEW_PENDING" if pending else "INDEPENDENT_QA_PENDING")
    reason = (f"{requests} human rejection notes require correction and re-review" if requests else
              (f"{pending} queue items unresolved" if pending else "Queue complete; independent quality evidence and release approval still required"))
    return Decision(status="REWORK" if requests else "HOLD", reason_code=code, reason=reason,
                    owner=ctx.config.get("owner", "UNASSIGNED"), scope="t07",
                    evidence=[str(ctx.out("redactions.csv")), str(ctx.out("review_state.json"))])


def run(ctx, mode="poc"):
    if ctx.track != "t07":
        raise ValueError("This module only supports T07")
    # An empty log means no human action yet; do not fabricate approvals.
    ctx.audit.path.touch(exist_ok=True)
    images = load(ctx)
    scored, machine = process(ctx, images, mode)
    if mode == "poc":
        queue = rank(ctx, scored)
        final = review(ctx, machine, queue, images)
        write_json(ctx.out("review_box_examples.json"), {str(r.item_id): final[r.item_id] for r in queue.itertuples()})
    else:
        final = machine
        write_json(ctx.out("review_state.json"), {"mode": mode, "human_corrections": False})
    export(ctx, images, final)
    if mode == "poc":
        render_review(ctx, queue)
    write_decision(decide(ctx), ctx.out("decision.json"), ctx.out("decision_log.csv"))
