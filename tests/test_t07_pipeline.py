"""T07 behaviour checks: export matches pixels, real review schema, replay safety."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.common.context import RunContext
from src.common.contact_sheet import _thumb_data_uri
from src.common.manifest import build_manifest, write_manifest
from src.common.io import sha256_file
from src.common.review import AuditLog
from src import t07_pipeline as pipeline
from src.t07_detectors import LocalDetectors
from tools.apply_t07_ai_corrections import apply as apply_ai_corrections


@pytest.fixture
def context(tmp_path, monkeypatch):
    root = tmp_path / "input"
    root.mkdir()
    image = np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(root / "sample.png"), image)
    pd.DataFrame([{"image_id": "sample", "path": "sample.png", "width": 64,
                   "height": 64, "split": "dev"}]).to_csv(root / "images.csv", index=False)
    write_manifest(build_manifest(root, ["images.csv", "sample.png"], "test", "CC0"), root / "manifest.csv")
    out = tmp_path / "out"
    out.mkdir()
    ctx = RunContext("t07", root, out, {"owner": "Test reviewer", "params": {"budget": 15, "padding": 0}},
                     42, np.random.default_rng(42), audit=AuditLog(out / "audit_log.jsonl"))

    class Detector:
        inventory = {kind: {"sha256": "test-model"} for kind in ("face", "plate")}

        def __init__(self, params):
            pass

        def detect(self, image):
            return [{"x": 10, "y": 10, "w": 16, "h": 16, "type": "face",
                     "confidence": 0.9, "source": "test"}]

    monkeypatch.setattr(pipeline, "LocalDetectors", Detector)
    monkeypatch.setattr(pipeline, "synthetic_proposals", lambda image: ([], 0))
    return ctx


def reviewed_file(ctx, **updates):
    queue = pd.read_csv(ctx.out("review_queue.csv"), dtype=str).fillna("")
    queue["reviewer"] = "Test reviewer"
    queue["reviewer_decision"] = "approve"
    queue["reviewed_at"] = "2026-11-14T16:00:00+07:00"
    for key, value in updates.items():
        queue[key] = value
    queue.to_csv(ctx.out("review_queue_reviewed.csv"), index=False)


def test_t07_exports_only_declared_region_and_holds_without_human(context):
    pipeline.run(context)
    pred = pd.read_csv(context.out("redactions.csv"))
    assert len(pred) == 1
    original = cv2.imread(str(context.input_dir / "sample.png"))
    redacted = cv2.imread(str(context.out("redacted/sample.png")))
    outside = np.ones((64, 64), dtype=bool)
    outside[10:26, 10:26] = False
    assert np.array_equal(original[outside], redacted[outside])
    assert not np.array_equal(original[10:26, 10:26], redacted[10:26, 10:26])
    decision = json.loads(context.out("decision.json").read_text())
    assert decision["status"] == "HOLD" and decision["reason_code"] == "HUMAN_REVIEW_PENDING"
    assert context.audit.read() == []  # Never invent a human action.


def test_precision_mode_uses_its_boxes_without_models_extra_padding_or_fake_review(context,monkeypatch):
    from src import t07_precision_pipeline as precise
    context.config['params']['synthetic_algorithm']='precision'
    def reject_pretrained(*args,**kwargs):raise AssertionError('Precision must not initialize photo detectors')
    monkeypatch.setattr(pipeline,'LocalDetectors',reject_pretrained)
    box=dict(x=7,y=9,w=11,h=13,type='face',confidence=.9,source='precision_test')
    monkeypatch.setattr(precise,'detect_details',lambda image:dict(boxes=[box],ambiguous_heads=0,
        algorithm='precision-dev-1',estimated_sigma=1.5))
    pipeline.run(context)
    output=pd.read_csv(context.out('redactions.csv'))
    assert tuple(output.iloc[0][['x','y','w','h']])==(7,9,11,13)
    original=cv2.imread(str(context.input_dir/'sample.png'));redacted=cv2.imread(str(context.out('redacted/sample.png')))
    outside=np.ones((64,64),bool);outside[9:22,7:18]=False
    assert np.array_equal(original[outside],redacted[outside])
    assert not np.array_equal(original[9:22,7:18],redacted[9:22,7:18])
    trace=json.loads(context.out('precision_trace.json').read_text())
    assert trace['sample']['route']=='2d' and trace['sample']['human_review_recorded'] is False
    assert context.stats['algorithm']=='precision-dev-1'
    assert context.audit.read()==[]


def test_human_can_replace_boxes_and_replay_without_duplicate_audit(context):
    pipeline.run(context)
    replacement = [{"x": 32, "y": 32, "w": 12, "h": 8, "type": "plate"}]
    reviewed_file(context, reviewer_decision="edit", override_value=json.dumps(replacement), review_note="Test replacement")
    pipeline.run(context)
    first_image = context.out("redacted/sample.png").read_bytes()
    first_csv = context.out("redactions.csv").read_bytes()
    events = context.audit.read()
    assert len(events) == 1 and events[0]["actor"] == "Test reviewer"
    assert events[0]["before"][0]["type"] == "face" and events[0]["after"][0]["type"] == "plate"
    pipeline.run(context)
    assert len(context.audit.read()) == 1
    assert context.out("redacted/sample.png").read_bytes() == first_image
    assert context.out("redactions.csv").read_bytes() == first_csv
    assert json.loads(context.out("decision.json").read_text())["status"] == "HOLD"  # Independent QA still missing.


def test_contact_sheet_shows_final_export_after_human_edit(context):
    pipeline.run(context)
    replacement = [{"x": 32, "y": 32, "w": 12, "h": 8, "type": "plate"}]
    reviewed_file(context, reviewer_decision="edit", override_value=json.dumps(replacement), review_note="Test edit")
    pipeline.run(context)
    page = context.out("contact_sheet.html").read_text(encoding="utf-8")
    assert _thumb_data_uri(context.out("redacted/sample.png"), 640) in page
    assert _thumb_data_uri(context.out("machine_images/sample.png"), 640) not in page
    queue = pd.read_csv(context.out("review_queue.csv"))
    assert Path(queue.iloc[0].evidence_path) == context.out("redacted/sample.png").resolve()


def test_restoring_previous_review_appends_transition_and_keeps_replay_idempotent(context):
    pipeline.run(context)
    a = json.dumps([{"x": 32, "y": 32, "w": 12, "h": 8, "type": "plate"}])
    b = json.dumps([{"x": 40, "y": 32, "w": 12, "h": 8, "type": "plate"}])
    for override, note in ((a, "Test A"), (b, "Test B"), (a, "Test A")):
        reviewed_file(context, reviewer_decision="edit", override_value=override, review_note=note)
        pipeline.run(context)
    events = context.audit.read()
    assert len(events) == 3
    assert events[-1]["before"][0]["x"] == 40
    assert events[-1]["after"][0]["x"] == 32
    assert events[-1]["extra"]["reapplied"] is True
    assert events[-1]["extra"]["reviewed_at"] == "2026-11-14T16:00:00+07:00"
    assert int(pd.read_csv(context.out("redactions.csv")).iloc[0].x) == 32
    pipeline.run(context)
    assert len(context.audit.read()) == 3


def test_approve_on_refreshed_page_keeps_displayed_correction(context):
    pipeline.run(context)
    replacement = [{"x": 32, "y": 32, "w": 12, "h": 8, "type": "plate"}]
    reviewed_file(context, reviewer_decision="edit", override_value=json.dumps(replacement), review_note="Test edit")
    pipeline.run(context)
    corrected_csv = context.out("redactions.csv").read_bytes()
    reviewed_file(context)
    pipeline.run(context)
    assert context.out("redactions.csv").read_bytes() == corrected_csv
    assert context.audit.read()[-1]["action"] == "review:approve"
    assert context.audit.read()[-1]["after"][0]["x"] == 32
    pipeline.run(context)
    assert len(context.audit.read()) == 2


def test_removing_review_records_machine_reset_without_inventing_reviewer(context):
    pipeline.run(context)
    replacement = [{"x": 32, "y": 32, "w": 12, "h": 8, "type": "plate"}]
    reviewed_file(context, reviewer_decision="edit", override_value=json.dumps(replacement), review_note="Test edit")
    pipeline.run(context)
    context.out("review_queue_reviewed.csv").unlink()
    pipeline.run(context)
    events = context.audit.read()
    assert len(events) == 2 and events[-1]["actor"] == "pipeline" and events[-1]["action"] == "review:reset"
    assert events[-1]["before"][0]["x"] == 32 and events[-1]["after"][0]["x"] == 10
    assert int(pd.read_csv(context.out("redactions.csv")).iloc[0].x) == 10
    pipeline.run(context)
    assert len(context.audit.read()) == 2


def test_reject_note_on_corrected_result_keeps_visible_boxes_and_requests_rework(context):
    pipeline.run(context)
    replacement = [{"x": 32, "y": 32, "w": 12, "h": 8, "type": "plate"}]
    reviewed_file(context, reviewer_decision="edit", override_value=json.dumps(replacement), review_note="Test edit")
    pipeline.run(context)
    corrected_csv = context.out("redactions.csv").read_bytes()
    reviewed_file(context, reviewer_decision="reject", override_value="", review_note="Correction still misses another face")
    pipeline.run(context)
    assert context.out("redactions.csv").read_bytes() == corrected_csv
    assert context.audit.read()[-1]["before"] == context.audit.read()[-1]["after"]
    assert json.loads(context.out("decision.json").read_text())["status"] == "REWORK"


def test_reject_note_without_coordinates_records_request_not_approval(context):
    pipeline.run(context)
    original_csv = context.out("redactions.csv").read_bytes()
    reviewed_file(context, reviewer_decision="reject", override_value="", review_note="Missed face: needs correction")
    pipeline.run(context)
    assert context.out("redactions.csv").read_bytes() == original_csv
    request = pd.read_csv(context.out("human_change_requests.csv")).iloc[0]
    assert request.reviewer == "Test reviewer" and request.status == "needs_correction_then_rereview"
    state = json.loads(context.out("review_state.json").read_text())
    assert state["n_completed"] == 0 and state["n_pending"] == 1 and state["n_correction_requests"] == 1
    decision = json.loads(context.out("decision.json").read_text())
    assert decision["status"] == "REWORK" and decision["reason_code"] == "HUMAN_CORRECTIONS_REQUIRED"
    pipeline.run(context)
    assert len(context.audit.read()) == 1


@pytest.mark.parametrize("updates, message", [
    ({"reviewer": ""}, "no `reviewer`"),
    ({"reason": "stale snapshot"}, "stale"),
    ({"reviewer_decision": "edit", "override_value": "not json", "review_note": "test"}, "JSON list"),
    ({"reviewer_decision": "edit", "override_value": '[{"x":60,"y":10,"w":20,"h":10,"type":"face"}]', "review_note": "test"}, "inside the image"),
    ({"reviewed_at": ""}, "reviewed_at"),
    ({"reviewer_decision": "reject", "override_value": "", "review_note": ""}, "review note"),
])
def test_invalid_human_review_fails_without_audit(context, updates, message):
    pipeline.run(context)
    reviewed_file(context, **updates)
    with pytest.raises(ValueError, match=message):
        pipeline.run(context)
    assert context.audit.read() == []


def test_missing_weights_fail_before_any_download(tmp_path):
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"face": {"path": str(tmp_path / "missing.onnx"), "sha256": "none"}}))
    with pytest.raises(FileNotFoundError, match="never downloads"):
        LocalDetectors({"model_inventory": str(inventory)})


def correction_inputs(context):
    pipeline.run(context)
    proposals = context.output_dir.parent / "proposals.json"
    summary = context.output_dir.parent / "ai_summary.json"
    proposals.write_text(json.dumps({"actor": "Codex (AI test)", "corrections": {
        "sample": [[32, 32, 12, 8, "plate"]]}}))
    summary.write_text(json.dumps({"evidence": [{"image_id": "sample",
        "original_sha256": sha256_file(context.input_dir / "sample.png"),
        "exported_sha256": sha256_file(context.out("redacted/sample.png"))}]}))
    return proposals, summary


def test_ai_correction_is_separate_attributed_and_replay_safe(context):
    proposals, summary = correction_inputs(context)
    original_csv = context.out("redactions.csv").read_bytes()
    out = context.output_dir.parent / "ai_corrected"
    for _ in range(2):
        apply_ai_corrections(context.input_dir, context.output_dir, out, proposals, summary)
    assert context.out("redactions.csv").read_bytes() == original_csv
    pred = pd.read_csv(out / "redactions.csv")
    assert set(pred.source) == {"ai_visual_correction"}
    events = AuditLog(out / "audit_log.jsonl").read()
    assert len(events) == 1 and events[0]["actor"] == "Codex (AI test)"
    assert events[0]["extra"]["human_review"] is False
    assert not context.out("review_queue_reviewed.csv").exists()


def test_ai_correction_rejects_stale_export_before_writing(context):
    proposals, summary = correction_inputs(context)
    payload = json.loads(summary.read_text())
    payload["evidence"][0]["exported_sha256"] = "0"*64
    summary.write_text(json.dumps(payload))
    out = context.output_dir.parent / "ai_corrected"
    with pytest.raises(ValueError, match="Reviewed export changed"):
        apply_ai_corrections(context.input_dir, context.output_dir, out, proposals, summary)
    assert not out.exists()


def test_auto_pipeline_exports_single_padding_and_route_without_human_event(context,monkeypatch):
    from src import t07_auto
    class Detector:
        photo=None
        def __init__(self,params):pass
        def detect_details(self,image):
            return dict(merged=[dict(x=20,y=20,w=20,h=20,type='face',confidence=.9,source='test')],
                        ambiguous_heads=0,algorithm='auto-native-1',routing={'route':'2d'},review_reasons=[])
    monkeypatch.setattr(t07_auto,'AutoDetectors',Detector)
    def forbidden(*a,**kw):raise AssertionError('old detector must not run')
    monkeypatch.setattr(pipeline,'LocalDetectors',forbidden)
    context.config['params'].update(detection_pipeline='auto',padding=.1)
    pipeline.run(context)
    prediction=pd.read_csv(context.out('redactions.csv')).iloc[0]
    assert [prediction[k] for k in ['x','y','w','h']]==[18,18,24,24]
    trace=json.loads(context.out('routing_trace.json').read_text())['sample']
    assert trace['routing']['route']=='2d' and trace['padding_applications']==1
    assert trace['bbox_calibration_applied'] is False
    assert context.audit.read()==[]
