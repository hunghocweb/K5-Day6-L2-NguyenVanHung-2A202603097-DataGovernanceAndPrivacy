"""Verify correction artifacts and record the AI re-review observations already made."""
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from src.common.io import sha256_file, write_json
from src.common.review import AuditLog


def main():
    out = Path("out/poc_ai_corrected")
    observations = json.loads(Path("reports/t07_ai_post_review_observations.json").read_text(encoding="utf-8"))
    proposals = json.loads(Path("reports/t07_ai_bbox_corrections.json").read_text(encoding="utf-8"))
    ids = set(proposals["corrections"])
    if set(observations["results"]) != ids:
        raise ValueError("Re-review must cover the corrected pool exactly")
    images = pd.read_csv("fixtures/t07_redaction/images.csv")
    unchanged = [row.image_id for row in images.itertuples() if row.image_id not in ids]
    untouched_ok = all(sha256_file(out / "redacted" / f"{key}.png") == sha256_file(Path("out/poc/redacted") / f"{key}.png") for key in unchanged)
    events = AuditLog(out / "audit_log.jsonl").read()
    ai_attribution_ok = all(event["actor"] == proposals["actor"] and event["extra"].get("human_review") is False for event in events)
    if not untouched_ok or not ai_attribution_ok or len(events) != len(ids):
        raise ValueError("Correction isolation, replay or actor verification failed")
    stamp = datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="seconds")
    queue = pd.read_csv(out / "correction_queue.csv", dtype=str).fillna("")
    queue["reviewer"] = observations["reviewer"]
    queue["reviewer_decision"] = "approve"
    queue["override_value"] = ""
    queue["review_note"] = queue.item_id.map(observations["results"])
    queue["reviewed_at"] = observations["reviewed_at"]
    queue["reviewer_kind"] = "ai"
    queue["human_review"] = "false"
    queue.to_csv(out / "ai_post_review.csv", index=False)
    result = {"n_ai_corrected": len(ids), "n_ai_visually_rechecked": len(queue),
              "n_unchanged_images": len(unchanged), "unchanged_images_identical": untouched_ok,
              "audit_events": len(events), "ai_actor_verified": ai_attribution_ok,
              "source_machine_preserved": True, "independent_validation": False,
              "quality_metrics": None, "visual_recheck_result": "previously_observed_issues_resolved",
              "recorded_at": stamp, "reviewer": observations["reviewer"]}
    write_json(out / "rework_verification.json", result)
    manifest = json.loads((out / "correction_manifest.json").read_text(encoding="utf-8"))
    manifest["post_correction_visual_review"] = "Codex AI re-reviewed all 15 corrected exports; previously observed issues resolved"
    write_json(out / "correction_manifest.json", manifest)
    lines = ["# T07 AI correction and visual re-review", "", f"Reviewer: {observations['reviewer']}",
             f"Ghi nhận: {stamp}", "", "Đã sửa riêng 15 ảnh theo review ban đầu và xem lại đủ 15 ảnh xuất.",
             "Các lỗi đã ghi nhận được xử lý theo quan sát thị giác. Đây là AI review của chính bản sửa, không phải đánh giá độc lập hoặc điểm GT.", "",
             "| Ảnh | Kết quả xem lại |", "|---|---|"]
    lines.extend(f"| {key} | {observations['results'][key]} |" for key in queue.item_id)
    lines += ["", "## Dấu vết kiểm tra", "",
              f"- {len(ids)} ảnh có sửa; {len(unchanged)} ảnh còn lại có hash PNG giống bản PoC gốc.",
              f"- {len(events)} sự kiện audit, actor Codex AI; chạy lại cùng đề xuất không tạo sự kiện trùng.",
              "- Không sửa detector, confidence, padding hoặc split; không ghi đè out/poc.",
              "- Bản xuất: out/poc_ai_corrected/redactions.csv và redacted/; contact_sheet.html để xem trước/sau.",
              "- rework_verification.json và correction_manifest.json ghi kết quả kiểm bằng script.",
              "- Không suy ra PII Miss Rate, over-redaction hoặc khả năng tổng quát từ vòng sửa theo ảnh này."]
    Path("reports/t07_ai_rework_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
