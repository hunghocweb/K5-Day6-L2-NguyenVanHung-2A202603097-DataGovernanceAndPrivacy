"""Local T07 visual reviewer. Run: python -m tools.t07_review_web."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import secrets
import threading
import zipfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import yaml

from src.common.io import sha256_file
from src.common.manifest import verify_manifest
from src.t07_pipeline import blur, digest

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src/web_review"


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_bytes(rows, columns):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


class Conflict(ValueError):
    pass


class ReviewStore:
    def __init__(self, root=ROOT, output=None):
        self.root = Path(root).resolve()
        self.output = Path(output or self.root / "out/web_review").resolve()
        allowed = self.root / "out"
        protected = [allowed / name for name in ("poc", "baseline", "poc_ai_corrected", "blur_all")]
        if not self.output.is_relative_to(allowed) or self.output == allowed or any(
                self.output.is_relative_to(path) for path in protected):
            raise ValueError("Web output must be a separate directory under out/")
        self.fixture = self.root / "fixtures/t07_redaction"
        config = self.root / "configs/lab.yaml"
        self.owner = (yaml.safe_load(config.read_text(encoding="utf-8")).get("owner", "G07")
                      if config.exists() else "G07")
        if verify_manifest(self.fixture / "manifest.csv"):
            raise ValueError("Fixture manifest failed; restore original inputs first")
        self.images = {row["image_id"]: row for row in read_csv(self.fixture / "images.csv")}
        self.queue = {row["item_id"]: row for row in read_csv(self.root / "out/poc/review_queue.csv")}
        self.base_dir = self.root / "out/poc_ai_corrected"
        if not (self.base_dir / "redactions.csv").is_file():
            self.base_dir = self.root / "out/poc"
        self.initial = {key: [] for key in self.images}
        self.source_hashes = {}
        for row in read_csv(self.base_dir / "redactions.csv"):
            self.initial[row["image_id"]].append({
                **{key: int(float(row[key])) for key in ("x", "y", "w", "h")},
                "type": row.get("type", "plate"),
            })
        for key, row in self.images.items():
            path = (self.fixture / row["path"]).resolve()
            if not path.is_relative_to(self.fixture) or not path.is_file():
                raise ValueError("Invalid source path")
            self.source_hashes[key] = sha256_file(path)
            self.initial[key] = self.validate_boxes(key, self.initial[key])
        self.fingerprint = digest({"manifest": sha256_file(self.fixture / "manifest.csv"),
                                   "predictions": sha256_file(self.base_dir / "redactions.csv"),
                                   "queue": sha256_file(self.root / "out/poc/review_queue.csv")})
        self.output.mkdir(parents=True, exist_ok=True)
        self.audit = self.output / "audit_log.jsonl"
        self.reviews = {}
        if self.audit.exists():
            for line in self.audit.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    event = json.loads(line)
                    if event["snapshot"] != self.fingerprint:
                        raise ValueError("Review source changed; use a new --output session directory")
                    self.reviews[event["item_id"]] = event
        self.lock = threading.RLock()
        self.token = secrets.token_urlsafe(32)

    def validate_boxes(self, key, boxes):
        if key not in self.images:
            raise ValueError("Unknown image")
        if not isinstance(boxes, list) or len(boxes) > 200:
            raise ValueError("Expected at most 200 regions")
        width, height = (int(self.images[key][name]) for name in ("width", "height"))
        clean = []
        for box in boxes:
            if not isinstance(box, dict) or box.get("type") not in {"face", "plate"}:
                raise ValueError("Region type must be face or plate")
            values = [box.get(name) for name in ("x", "y", "w", "h")]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   or int(v) != v for v in values):
                raise ValueError("Region coordinates must be finite integers")
            x, y, w, h = map(int, values)
            if x < 0 or y < 0 or w < 1 or h < 1 or x+w > width or y+h > height:
                raise ValueError("Region outside image bounds")
            clean.append(dict(x=x, y=y, w=w, h=h, type=box["type"]))
        return clean

    def boxes(self, key):
        return self.reviews[key]["after"] if key in self.reviews else self.initial[key]

    def revision(self, key):
        return digest(self.reviews.get(key, {"snapshot": self.fingerprint, "boxes": self.initial[key]}))

    def state(self):
        with self.lock:
            items = []
            for key, row in self.images.items():
                queue, review = self.queue.get(key, {}), self.reviews.get(key, {})
                items.append({"id": key, "split": row["split"], "width": int(row["width"]),
                              "height": int(row["height"]), "rank": int(queue["rank"]) if queue else None,
                              "score": float(queue["score"]) if queue else None,
                              "reason": queue.get("reason", ""), "boxes": self.boxes(key),
                              "revision": self.revision(key), "decision": review.get("decision", ""),
                              "reviewer": review.get("actor", ""), "note": review.get("reason", ""),
                              "reviewed_at": review.get("ts", "")})
            items.sort(key=lambda item: (item["rank"] or 999, item["id"]))
            return {"items": items, "token": self.token, "source": self.base_dir.name,
                    "output": str(self.output.relative_to(self.root)), "snapshot": self.fingerprint,
                    "quality_metrics": None, "gate": self.gate()}

    def gate(self):
        reviewed = [self.reviews.get(key, {}) for key in self.queue]
        completed = sum(event.get("decision") in {"approve", "edit"} for event in reviewed)
        rejected = sum(event.get("decision") == "reject" for event in self.reviews.values())
        return {"status": "REWORK" if rejected else "HOLD", "completed": completed,
                "budget": len(self.queue), "reason_code": "REVIEW_REJECTED" if rejected else
                "HUMAN_REVIEW_PENDING" if completed < len(self.queue) else "INDEPENDENT_QA_PENDING",
                "owner": self.owner, "scope": "t07_web_review",
                "reason": "Có ảnh cần sửa tiếp" if rejected else "Queue còn ảnh chưa xử lý" if
                completed < len(self.queue) else "Cần đánh giá chất lượng độc lập trước khi release"}

    def source_image(self, key):
        if key not in self.images:
            raise ValueError("Unknown image")
        path = self.fixture / self.images[key]["path"]
        if sha256_file(path) != self.source_hashes[key]:
            raise ValueError("Ảnh nguồn đã thay đổi. Khôi phục dữ liệu và khởi động lại server.")
        return path

    def render(self, key, boxes):
        image = cv2.imread(str(self.source_image(key)))
        ok, encoded = cv2.imencode(".png", blur(image, self.validate_boxes(key, boxes)))
        if not ok:
            raise ValueError("PNG export failed")
        return encoded.tobytes()

    def save(self, payload):
        key = payload.get("item_id")
        if key not in self.images:
            raise ValueError("Unknown image")
        actor, note = payload.get("reviewer", ""), payload.get("note", "")
        if not isinstance(actor, str) or not actor.strip() or len(actor) > 160:
            raise ValueError("Nhập tên reviewer trước khi lưu")
        if not isinstance(note, str) or len(note) > 4000:
            raise ValueError("Review note too long")
        decision = payload.get("decision")
        if decision not in {"approve", "edit", "reject", "skip", "escalate"}:
            raise ValueError("Chọn quyết định trước khi lưu")
        if decision in {"reject", "escalate"} and not note.strip():
            raise ValueError("Nhập lý do cần sửa hoặc chuyển xử lý")
        after = self.validate_boxes(key, payload.get("boxes"))
        with self.lock:
            if payload.get("revision") != self.revision(key):
                raise Conflict("Ảnh đã được lưu ở tab khác. Tải lại trang để nhận bản mới.")
            before = self.boxes(key)
            event = {"ts": datetime.now(timezone.utc).isoformat(), "actor": actor.strip(),
                     "action": f"human_review:{decision}", "item_id": key, "decision": decision,
                     "before": before, "after": after, "reason": note.strip(),
                     "snapshot": self.fingerprint, "source_image_sha256": sha256_file(self.source_image(key)),
                     "extra": {"reviewer_kind": "human", "human_review": True, "interface": "local_web"}}
            # The append-only journal is authoritative; projections can be rebuilt after restart.
            with self.audit.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                handle.flush()
            self.reviews[key] = event
            target = self.output / "redacted" / f"{key}.png"
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(self.render(key, after))
            self.projections()
            return self.state()

    def projections(self):
        predictions = [{"image_id": key, **box} for key in self.images for box in self.boxes(key)]
        reviews = [{"item_id": key, "reviewer": event["actor"], "reviewer_decision": event["decision"],
                    "override_value": json.dumps(event["after"], ensure_ascii=False),
                    "review_note": event["reason"], "reviewed_at": event["ts"],
                    "split": self.images[key]["split"], "snapshot": self.fingerprint}
                   for key, event in self.reviews.items()]
        (self.output / "redactions.csv").write_bytes(csv_bytes(predictions, ["image_id", "x", "y", "w", "h", "type"]))
        (self.output / "review_queue_reviewed.csv").write_bytes(csv_bytes(reviews,
            ["item_id", "reviewer", "reviewer_decision", "override_value", "review_note", "reviewed_at", "split", "snapshot"]))
        (self.output / "decision.json").write_text(json.dumps(self.gate(), indent=2), encoding="utf-8")

    def export(self):
        with self.lock:
            self.projections()
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                for key in self.images:
                    archive.writestr(f"redacted/{key}.png", self.render(key, self.boxes(key)))
                for name in ("redactions.csv", "review_queue_reviewed.csv", "decision.json", "audit_log.jsonl"):
                    path = self.output / name
                    archive.writestr(name, path.read_bytes() if path.exists() else b"")
                archive.writestr("session.json", json.dumps({"snapshot": self.fingerprint,
                    "initial_source": str(self.base_dir.relative_to(self.root)),
                    "n_images": len(self.images), "gate": self.gate(), "quality_metrics": None}))
            result = buffer.getvalue()
            (self.output / "t07_review_export.zip").write_bytes(result)
            return result


def make_handler(store):
    upload_lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, body, content_type="application/json; charset=utf-8", **headers):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in headers.items():
                self.send_header(key.replace("_", "-"), value)
            self.end_headers()
            self.wfile.write(body)

        def valid_host(self):
            return self.headers.get("Host") in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

        def do_GET(self):
            if not self.valid_host():
                return self.respond(403, b'{}')
            route = urlparse(self.path)
            try:
                if route.path in {"/api/walk/status", "/api/walk/file"}:
                    from tools.t07_walkthrough import folder_for
                    query=parse_qs(route.query)
                    folder=folder_for(query.get('job',[''])[0])
                    if route.path.endswith('/status'):
                        return self.respond(200,(folder/'status.json').read_bytes())
                    name=query.get('name',[''])[0]
                    if name not in {'source.png','letterbox.png','tiles.png','detections.png','merged.png','padding.png','result.png','result.zip'}:
                        raise ValueError('Unknown artifact')
                    mime='application/zip' if name.endswith('.zip') else 'image/png'
                    headers={'Content_Disposition':f'attachment; filename="t07_{name}"'} if query.get('download') else {}
                    return self.respond(200,(folder/name).read_bytes(),mime,**headers)
                if route.path == "/api/state":
                    return self.respond(200, json.dumps(store.state(), ensure_ascii=False).encode())
                if route.path == "/api/rework":
                    path=store.root/'reports/t07_reject_rework.json'
                    proposals=json.loads(path.read_text(encoding='utf-8')).get('proposals',{}) if path.exists() else {}
                    valid={key:value for key,value in proposals.items() if key in store.images
                           and value.get('review_revision')==store.revision(key)
                           and value.get('source_image_sha256')==sha256_file(store.source_image(key))}
                    for key,value in valid.items():store.validate_boxes(key,value['boxes'])
                    return self.respond(200,json.dumps(valid,ensure_ascii=False).encode())
                if route.path == "/api/export":
                    return self.respond(200, store.export(), "application/zip",
                                        Content_Disposition='attachment; filename="t07_review_export.zip"')
                if route.path == "/api/image":
                    query = parse_qs(route.query)
                    key, mode = query.get("id", [""])[0], query.get("mode", ["source"])[0]
                    store.source_image(key)  # allow only manifest IDs; never arbitrary filesystem paths
                    if mode == "source":
                        path = store.source_image(key)
                    elif mode in {"baseline", "poc", "poc_ai_corrected"}:
                        path = store.root / "out" / mode / "redacted" / f"{key}.png"
                    else:
                        raise ValueError("Unknown image mode")
                    return self.respond(200, path.read_bytes(), "image/jpeg" if mode == "source" else "image/png")
                assets = {"/": ("index.html", "text/html; charset=utf-8"),
                          "/walkthrough": ("walkthrough.html", "text/html; charset=utf-8"),
                          "/walkthrough.js": ("walkthrough.js", "text/javascript; charset=utf-8"),
                          "/walkthrough.css": ("walkthrough.css", "text/css; charset=utf-8"),
                          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                          "/style.css": ("style.css", "text/css; charset=utf-8")}
                if route.path in assets:
                    name, mime = assets[route.path]
                    return self.respond(200, (ASSETS / name).read_bytes(), mime)
                self.respond(404, b'{"error":"Not found"}')
            except (ValueError, FileNotFoundError) as exc:
                self.respond(404, json.dumps({"error": str(exc)}).encode())

        def do_POST(self):
            origin = self.headers.get("Origin")
            if not self.valid_host() or self.headers.get("X-Review-Token") != store.token or (
                    origin and origin != f"http://{self.headers.get('Host')}"):
                return self.respond(403, b'{"error":"Invalid local session"}')
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if urlparse(self.path).path == '/api/walk/upload':
                    from tools.t07_walkthrough import start,MAX_UPLOAD
                    if not 0 < length <= MAX_UPLOAD:raise ValueError('Chọn ảnh dưới 15 MB.')
                    query=parse_qs(urlparse(self.path).query)
                    raw=self.rfile.read(length)
                    with upload_lock:
                        result=start(raw,query.get('mode',['auto'])[0],float(query.get('padding',['0'])[0]))
                    return self.respond(202,json.dumps(result).encode())
                if length < 1 or length > 200_000:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Expected JSON object")
                if self.path == "/api/preview":
                    return self.respond(200, store.render(payload.get("item_id"), payload.get("boxes")), "image/png")
                if self.path != "/api/review":
                    return self.respond(404, b'{}')
                self.respond(200, json.dumps(store.save(payload), ensure_ascii=False).encode())
            except (ValueError, TypeError, KeyError) as exc:
                self.respond(409 if isinstance(exc, Conflict) else 400,
                             json.dumps({"error": str(exc)}, ensure_ascii=False).encode())
            except OSError:
                self.respond(500, json.dumps({"error": "Không lưu hoặc khởi động được xử lý local. Xem log server."}, ensure_ascii=False).encode())

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", type=Path, default=ROOT / "out/web_review")
    args = parser.parse_args()
    store = ReviewStore(output=args.output)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store))
    print(f"T07 review: http://127.0.0.1:{server.server_port} | output={store.output}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
