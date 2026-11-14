"""Scope the existing submission to its recorded original 15-item review queue."""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile


def scope_submission(root: Path) -> dict:
    reports = root / "reports"
    bundle = reports / "evidence/review15/results.zip"
    queue_bytes = (reports / "review_queue.csv").read_bytes()
    queue = list(csv.DictReader(io.StringIO(queue_bytes.decode("utf-8-sig"))))
    if (len(queue) != 15 or len({r["item_id"] for r in queue}) != 15
            or any(r["reviewer_decision"] not in {"approve", "edit"}
                   or not r["reviewer"].strip() for r in queue)):
        raise ValueError("Need the original 15 completed recorded reviews")
    with zipfile.ZipFile(bundle) as archive:
        entries = [(info, archive.read(info.filename)) for info in archive.infolist()]
    original = {info.filename: data for info, data in entries}
    if sum(name.startswith("redacted/") and name.endswith(".png") for name in original) != 60:
        raise ValueError("Need the original 60-image bundle")
    if original["review_queue_reviewed.csv"] != queue_bytes:
        raise ValueError("Bundle review queue differs from recorded queue")
    manifest = json.loads(original["export_manifest.json"])
    if (manifest["queue_ids"] != [r["item_id"] for r in queue]
            or manifest["unchanged_outside_queue"] != 45
            or manifest["predictions_sha256"] != hashlib.sha256(original["redactions.csv"]).hexdigest()):
        raise ValueError("Original queue or prediction checksum mismatch")
    manifest.pop("extra_reviews", None)
    manifest.update(review_scope="original_top15", scope_updated_at=datetime.now().astimezone().isoformat())
    decision = dict(status="HOLD", reason_code="INDEPENDENT_QA_PENDING",
                    owner=json.loads(original["decision.json"])["owner"],
                    scope="original_top15", completed=15, budget=15,
                    n_images=60, review_budget=15, queue_rejected_ids=[],
                    official_quality_metrics=None, extra_reviews_applied=False)
    encode = lambda value: (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    updates = {"decision.json": encode(decision), "export_manifest.json": encode(manifest),
               "README.txt": ("T07 G07 original review15 submission\n"
                              "60 synthetic images; seed42; 14 approve + 1 edit.\n"
                              "45 outside-queue images retain original machine output.\n"
                              "Status: HOLD / INDEPENDENT_QA_PENDING.\n").encode("utf-8")}
    backup = Path(tempfile.mkdtemp(prefix="t07_before_review15_scope_"))
    (backup / "results.zip").write_bytes(bundle.read_bytes())
    (backup / "decision_log.csv").write_bytes((reports / "decision_log.csv").read_bytes())
    temporary = bundle.with_suffix(".queue15.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for info, data in entries:
            if info.filename in updates:
                archive.writestr(info.filename, updates[info.filename])
            else:
                archive.writestr(info, data)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("Invalid scoped bundle")
        for name, data in original.items():
            if name not in updates and archive.read(name) != data:
                raise ValueError(f"Original artifact changed: {name}")
    os.replace(temporary, bundle)
    (reports / "evidence/review15_manifest.json").write_bytes(updates["export_manifest.json"])
    (reports / "evidence/review15/decision.json").write_bytes(updates["decision.json"])
    stream = io.StringIO(newline="")
    columns = ["status", "reason_code", "owner", "scope", "completed", "budget", "queue_rejected_ids"]
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerow({key: decision[key] for key in columns})
    (reports / "decision_log.csv").write_bytes(stream.getvalue().encode("utf-8"))
    return {"scope": "original_top15", "completed": 15, "preserved_images": 60,
            "preserved_journals": True, "backup": str(backup)}


if __name__ == "__main__":
    print(json.dumps(scope_submission(Path(__file__).resolve().parents[1])))
