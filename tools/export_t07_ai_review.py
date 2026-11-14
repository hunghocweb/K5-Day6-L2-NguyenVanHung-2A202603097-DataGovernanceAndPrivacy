"""Export recorded AI visual observations separately from real-human review evidence."""
import json
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from src.common.io import sha256_file
from src.common.review import load_review_decisions


def main():
    observations = json.loads(Path("reports/t07_ai_review_observations.json").read_text(encoding="utf-8"))
    queue = pd.read_csv("out/poc/review_queue.csv", dtype=str).fillna("")
    findings = {row["image_id"]: row for row in observations["findings"]}
    if len(findings) != 15 or set(queue.item_id) != set(findings):
        raise ValueError("Observations must match exactly the 15 images visually inspected")
    stamp = datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="seconds")
    review = queue.copy()
    review["reviewer"] = observations["reviewer"]
    review["reviewer_decision"] = review.item_id.map(lambda key: findings[key]["decision"])
    review["review_note"] = review.item_id.map(lambda key: findings[key]["note"])
    review["override_value"] = ""
    review["reviewed_at"] = stamp
    review["reviewer_kind"] = "ai"
    review["human_review"] = "false"
    folder = Path("out/ai_review")
    folder.mkdir(parents=True, exist_ok=True)
    review_path = folder / "review_queue_reviewed.csv"
    review.to_csv(review_path, index=False)
    validated = load_review_decisions(review_path)
    if len(validated) != 15 or set(validated.reviewer) != {observations["reviewer"]}:
        raise ValueError("Export did not preserve AI reviewer attribution")
    counts = Counter(review.reviewer_decision)
    tags = Counter(tag for row in findings.values() for tag in row["tags"])
    evidence = []
    for row in queue.itertuples(index=False):
        evidence.append({"image_id": row.item_id, "split": row.slice, "snapshot": row.reason,
                         "original_path": row.original_path, "original_sha256": sha256_file(row.original_path),
                         "exported_path": row.evidence_path, "exported_sha256": sha256_file(row.evidence_path),
                         "decision": findings[row.item_id]["decision"], "tags": findings[row.item_id]["tags"]})
    summary = {"reviewer": observations["reviewer"], "reviewer_kind": "ai", "human_review": False,
               "recorded_at": stamp, "n_reviewed": len(review), "decisions": dict(counts),
               "images_by_issue_tag": dict(tags), "recommended_decision": "REWORK",
               "method": observations["method"], "evidence": evidence, "quality_metrics": None,
               "scope": "Priority queue only; visual observations are not hidden-GT metrics",
               "production_human_review_csv_written": False}
    (folder / "review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# T07 G07 AI visual review", "", f"Reviewer: {observations['reviewer']}", f"Ghi nhận: {stamp}", "",
             f"Đã xem trực tiếp {len(review)} cặp ảnh nguồn và ảnh xuất trong queue ưu tiên. Kết quả: {counts['reject']} reject; đề xuất REWORK.", "",
             "Quan sát từ ảnh hiển thị; không dùng GT ẩn và không suy ra PII Miss Rate hoặc over-redaction từ số quyết định.", "",
             "## Kết quả theo ảnh", "", "| Rank | Ảnh | Split | Quyết định | Lý do |", "|---|---|---|---|---|"]
    for row in queue.itertuples(index=False):
        finding = findings[row.item_id]
        lines.append(f"| {row.rank} | {row.item_id} | {row.slice} | {finding['decision']} | {finding['note']} |")
    lines += ["", "## Nhóm lỗi cần làm lại", ""]
    labels = {"partial_plate_redaction": "Biển số còn ký tự ngoài vùng che", "missed_faces": "Mặt bị bỏ sót",
              "unredacted_plate": "Biển số chưa được che hoặc chưa có vùng che rõ ràng",
              "non_pii_redaction": "Vùng che nằm trên kính xe"}
    for tag, count in tags.items():
        lines.append(f"- {labels[tag]}: {count} ảnh trong queue. Một ảnh có thể có nhiều nhóm lỗi.")
    lines += ["", "## Hành động tiếp theo", "",
              "- Giữ nguyên baseline, split và detector đã freeze. Không dùng ghi chú của ảnh eval để tune tham số chung.",
              "- Với từng ảnh reject, lập đề xuất sửa vùng bị sót/che thiếu/che nhầm, lưu phiên bản trước và sau cùng tác giả thực hiện.",
              "- Ưu tiên mặt còn thấy trong ảnh mờ/tối và phần chữ biển số lộ ngoài bbox; đối chiếu định nghĩa vùng GT khi đánh giá metric.",
              "", "## File kết quả", "",
              "- CSV quyết định: `out/ai_review/review_queue_reviewed.csv`.",
              "- Tóm tắt và hash ảnh đã xem: `out/ai_review/review_summary.json`.",
              "- Ghi chú nguồn: `reports/t07_ai_review_observations.json`.",
              "- Các quyết định ghi dưới danh tính AI reviewer, tách riêng khỏi file reviewer người thật của pipeline."]
    Path("reports/t07_ai_visual_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"n_reviewed": len(review), "decisions": dict(counts), "issue_counts": dict(tags),
                      "report": "reports/t07_ai_visual_review.md", "csv": str(review_path),
                      "reviewer": observations["reviewer"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
