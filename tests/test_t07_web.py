"""Exercise real saved state, pixels and HTTP boundaries without touching real reviews."""
import json
import shutil
import threading
import urllib.error
import urllib.request
import zipfile
import io
from http.server import ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from tools.t07_review_web import ROOT, Conflict, ReviewStore, csv_bytes, read_csv, make_handler


@pytest.fixture
def store(tmp_path):
    shutil.copytree(ROOT / "fixtures/t07_redaction", tmp_path / "fixtures/t07_redaction")
    images = read_csv(tmp_path / "fixtures/t07_redaction/images.csv")
    predictions = [{"image_id": row["image_id"], "x": 20, "y": 20, "w": 30, "h": 30, "type": "face"}
                   for row in images]
    for name in ("poc", "poc_ai_corrected"):
        target = tmp_path / "out" / name
        target.mkdir(parents=True)
        (target / "redactions.csv").write_bytes(csv_bytes(predictions, ["image_id", "x", "y", "w", "h", "type"]))
    queue = [{"item_id": row["image_id"], "rank": n+1, "score": 15-n, "reason": "test queue"}
             for n, row in enumerate(images[:15])]
    (tmp_path / "out/poc/review_queue.csv").write_bytes(csv_bytes(queue, ["item_id", "rank", "score", "reason"]))
    return ReviewStore(tmp_path)


def payload(store, key="pii_10"):
    return {"item_id": key, "revision": store.revision(key), "reviewer": "Test reviewer",
            "decision": "edit", "note": "Regression test", "boxes": [
                {"x": 100, "y": 100, "w": 30, "h": 20, "type": "face"}]}


def test_save_export_and_restart_preserve_pixels_history_and_initial_run(store):
    key = "pii_10"
    initial = (store.base_dir / "redactions.csv").read_bytes()
    request = payload(store, key)
    state = store.save(request)
    assert next(row for row in state["items"] if row["id"] == key)["reviewer"] == "Test reviewer"
    with pytest.raises(Conflict):
        store.save(request)  # stale tab cannot overwrite newer edits
    restarted = ReviewStore(store.root)
    assert restarted.boxes(key) == request["boxes"]
    request2 = payload(restarted, key)
    request2["decision"] = "approve"
    restarted.save(request2)
    events = [json.loads(line) for line in restarted.audit.read_text(encoding="utf-8").splitlines()]
    assert len(events) == 2
    assert events[1]["before"] == events[0]["after"]
    assert events[1]["actor"] == "Test reviewer"
    assert (store.base_dir / "redactions.csv").read_bytes() == initial
    with zipfile.ZipFile(io.BytesIO(restarted.export())) as archive:
        assert len([name for name in archive.namelist() if name.startswith("redacted/")]) == 60
        image = cv2.imdecode(np.frombuffer(archive.read(f"redacted/{key}.png"), np.uint8), cv2.IMREAD_COLOR)
        source = cv2.imread(str(store.source_image(key)))
        outside = np.ones(source.shape[:2], dtype=bool)
        outside[100:120, 100:130] = False
        assert np.array_equal(image[outside], source[outside])
        assert json.loads(archive.read("decision.json"))["status"] != "GO"


@pytest.mark.parametrize("change", [
    {"reviewer": ""}, {"decision": "reject", "note": ""},
    {"boxes": [{"x": -1, "y": 0, "w": 20, "h": 20, "type": "face"}]},
    {"boxes": [{"x": 1.5, "y": 0, "w": 20, "h": 20, "type": "face"}]},
    {"boxes": [{"x": 630, "y": 0, "w": 20, "h": 20, "type": "plate"}]},
])
def test_invalid_review_never_appends_event(store, change):
    request = payload(store) | change
    with pytest.raises(ValueError):
        store.save(request)
    assert not store.audit.exists()


def test_rejection_outside_queue_requires_rework(store):
    key=next(key for key in store.images if key not in store.queue)
    request=payload(store,key) | {"decision":"reject","note":"Visible missed region outside queue"}
    store.save(request)
    assert store.gate()["status"]=="REWORK"
    assert store.gate()["reason_code"]=="REVIEW_REJECTED"
    request=payload(store,key) | {"decision":"edit"}
    store.save(request)
    assert store.gate()["status"]=="HOLD"


def test_rework_proposal_is_read_only_and_expires_after_review(store):
    from src.common.io import sha256_file
    folder=store.root/'reports';folder.mkdir()
    key='pii_10'
    proposal={key:{'boxes':[{'x':10,'y':10,'w':20,'h':20,'type':'face'}],
                   'review_revision':store.revision(key),
                   'source_image_sha256':sha256_file(store.source_image(key))}}
    (folder/'t07_reject_rework.json').write_text(json.dumps({'proposals':proposal}),encoding='utf-8')
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(store))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    url=f'http://127.0.0.1:{server.server_port}/api/rework'
    try:
        with urllib.request.urlopen(url) as response:assert json.load(response)==proposal
        assert not store.audit.exists()  # viewing a proposed correction is not a human action
        store.save(payload(store,key))
        with urllib.request.urlopen(url) as response:assert json.load(response)=={}
    finally:
        server.shutdown();server.server_close()


def test_http_rejects_foreign_write_and_paths_but_accepts_local_session(store):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/api/state") as response:
            token = json.load(response)["token"]
        raw = json.dumps(payload(store)).encode()
        request = urllib.request.Request(base + "/api/review", data=raw,
                                         headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(request)
        assert exc.value.code == 403
        request.add_header("X-Review-Token", token)
        request.add_header("Origin", "https://example.invalid")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(request)
        assert exc.value.code == 403
        request.remove_header("Origin")
        with urllib.request.urlopen(request) as response:
            assert response.status == 200
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(base + "/api/image?id=../../README&mode=source")
        assert exc.value.code == 404
        bad_upload=urllib.request.Request(base+'/api/walk/upload',data=b'not an image',
                                          headers={'X-Review-Token':token})
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(bad_upload)
        assert exc.value.code==400
        bad_upload=urllib.request.Request(base+'/api/walk/upload',data=b'not an image')
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(bad_upload)
        assert exc.value.code==403
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(base+'/api/walk/file?job=../../README&name=result.png')
        assert exc.value.code==404
    finally:
        server.shutdown()
        server.server_close()
