"""Build submission prose from actual run artifacts without fabricating GT metrics."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    result = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", *args], cwd=ROOT,
                            text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else "not available"


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main():
    cfg = yaml.safe_load((ROOT / "configs/lab.yaml").read_text(encoding="utf-8"))
    runs = read("reports/t07_run_summary.json")["runs"]
    validation = read("reports/t07_validation.json")
    review = read("out/ai_review/review_summary.json")
    rework = read("out/poc_ai_corrected/rework_verification.json")
    info = read("out/poc/run_info.json")
    freeze = read("reports/t07_protocol_freeze.json")
    commit = git("rev-parse", "HEAD")
    name, email = git("config", "--get", "user.name"), git("config", "--get", "user.email")
    remote = git("remote", "get-url", "origin")
    if "://" in remote:
        url = urlsplit(remote)
        remote = urlunsplit((url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, "", ""))
    table = ["| Run | N ảnh | N bbox | Runtime tổng (s) | s/100 ảnh | Metric chất lượng GT |",
             "|---|---:|---:|---:|---:|---|"]
    for key in ("baseline", "poc", "blur_all"):
        run = runs[key]
        table.append(f"| {key} | {run['n_images']} | {run['n_regions']} | {run['runtime_seconds']} | {run['runtime_per_100_images_seconds']:.3f} | Chưa đo |")
    report = f"""# G07 T07 Face and Plate Redaction

Nguyễn Văn Hùng · MSSV 202603097 · owner/reviewer theo cấu hình. Repo: {remote}

## 1 Problem statement và phạm vi

Pipeline hỗ trợ che mặt/biển số trước khi xuất dữ liệu ảnh, ưu tiên trường hợp rủi ro cho review và lưu bằng chứng thay đổi. Bỏ sót có thể để lộ thông tin định danh; che nhầm làm mất tín hiệu hữu ích. Bản lab dùng dữ liệu synthetic, không phải chứng nhận ẩn danh hóa hoặc tuân thủ pháp luật.

Đầu vào là images.csv, ảnh và manifest; đầu ra gồm redactions.csv (image_id,x,y,w,h), PNG đã che, queue B=15, audit và quyết định có reason_code/owner. Policy hiện tại: chưa review → HOLD; reject có lý do → REWORK; chưa có chứng cứ chất lượng độc lập → không tự GO. Owner là {cfg['owner']}; ghi owner không thay thế thao tác reviewer.

## 2 Dữ liệu và provenance

| Split | N ảnh | Kích thước | Nguồn | License |
|---|---:|---|---|---|
| dev | 36 | 640×360 | synthetic:tools/make_fixtures.py, seed42 | CC0-1.0 theo manifest |
| eval | 24 | 640×360 | cùng fixture, split giữ nguyên | CC0-1.0 theo manifest |

Pipeline verify SHA256 trước đọc ảnh. Ground truth và generator thuộc overlay giảng viên, không có trong repo học viên. Không dùng dữ liệu thật hoặc upload PII. Chỉ mở dev để thiết kế trước freeze; lần chạy all gồm eval sau freeze dùng tham số cố định. Bản sửa theo ảnh từ AI được lưu riêng, không dùng để tune detector hoặc coi là một bộ GT độc lập.

## 3 Thiết kế và thành phần offline

Load/provenance → YuNet mặt + YOLOv9-t plate local → rule synthetic bổ sung → rank risk → review/override → export/audit/decision.

- YuNet 2023mar: MIT; input 320×320; model và checksum trong reports/t07_models.json.
- YOLOv9-t 640 license plate: ONNX CPU, open-image-models 0.6.0, onnxruntime 1.32.0. Repo upstream MIT, quyền đối với weights chưa được tác giả kết luận đầy đủ; không khẳng định eligibility thương mại.
- PoC dùng rule màu da/hình dạng đầu + dấu hiệu hai mắt và vùng plate hình học. Đây là heuristic trên synthetic, không phải model head pretrained. YuNet tạo 0 bbox mặt trong dev probe; YOLO tạo 29 bbox plate trên 36 dev. Các số này là detection count, không phải recall.
- Tham số detector/rank đã freeze; thiếu weights hoặc sai hash phải fail rõ, không tự download khi chạy offline.

## 4 Baseline và kết quả vận hành

Baseline là detector mặc định + blur sigma12, không correction. PoC cùng detector confidence, thêm proposals/risk và padding. Blur-all báo riêng như đối chứng bắt buộc.

{chr(10).join(table)}

Bảng do tools.summarize_t07 sinh từ run_info.json. Runtime gồm xuất PNG/contact sheet và ảnh hưởng tải máy; không phải benchmark inference cô lập. N bbox tăng không chứng minh recall tăng hay guardrail đạt.

Nhánh AI correction sửa {rework['n_ai_corrected']} ảnh từ queue, xem lại {rework['n_ai_visually_rechecked']} ảnh; {rework['n_unchanged_images']} ảnh ngoài queue có hash giữ nguyên. Nhánh này có actor AI, tách khỏi kết quả detector gốc. Không báo mức tăng chất lượng model từ việc AI tự sửa và tự xem lại.

## 5 Protocol và metric

Protocol freeze ghi tại {freeze['frozen_at']}, hash file ở reports/t07_protocol_freeze.json. Seed42, B15, face confidence0.90, plate0.25, padding0.03; chỉnh trên dev trước đánh giá.

Primary PII Miss Rate = số vùng must_redact có coverage dưới0.90 / tổng vùng must_redact. Coverage là hợp pixel vùng che trong vùng GT, không thay bằng IoU0.5. Guardrail over-redaction = pixel che ngoài hợp vùng GT / tổng pixel che ≤0.30. Báo face/plate, condition, kích thước vùng; dự kiến bootstrap theo ảnh khi có nhãn độc lập.

Hiện chưa có GT chính thức hoặc nhãn kiểm chứng độc lập nội bộ. PII Miss Rate, over-redaction, Recall@B, size slices và CI chưa đo. reports/metrics_baseline.json, metrics_poc.json và table1 chưa tạo; không điền số bằng tay. AI bbox sửa không được dùng làm nhãn để tự chấm chính bản sửa đó.

## 6 Hai failure case có bằng chứng

**pii_10 (dev), biển số che thiếu ký tự:** bản máy che nền biển nhưng còn đuôi chữ/số ngoài mép phải. Bbox YOLO và padding cố định chưa phủ đủ chuỗi nhìn thấy; sửa riêng theo ảnh mở rộng vùng che. Xem fixtures/t07_redaction/images/pii_10.jpg, out/poc/redacted/pii_10.png và out/poc_ai_corrected/redacted/pii_10.png. Kết luận này là quan sát thị giác; không sửa GT hoặc suy ra metric chính thức.

**pii_40 (dev), mặt mờ bị sót và kính xe bị che nhầm:** YuNet không phát hiện được mặt synthetic, eye-rule khó tìm dấu hiệu trên ảnh mờ; plate-rule nhận vùng kính có tương phản thành plate. Bản sửa bổ sung bốn vùng mặt và hai plate, bỏ vùng kính. Bằng chứng ba phiên bản cùng image_id như trên. Detector không bị tune theo các ảnh eval đã review.

## 7 Review và audit

Queue gồm15 ảnh rủi ro cao, thứ tự và snapshot cố định. Codex trực tiếp xem ảnh trước/sau và ghi {review['decisions'].get('reject', 0)} reject; reviewer_kind=ai. Kết quả theo ảnh tại reports/t07_ai_visual_review.md. Review và sửa AI không được gán cho Nguyễn Văn Hùng hoặc coi là reviewer người thật đã hành động.

Mỗi thay đổi có before/after, actor, timestamp và hash nguồn. Contact sheet lấy ảnh cuối sau export. A→A không lặp audit; A→B→A ghi chuyển trạng thái mới; bỏ review file ghi reset actor pipeline. Reject chỉ có ghi chú không đoán bbox và không thành approve. Test reviewer giả không tính là review thật.

Bản sửa AI có15 sự kiện audit, giữ45 ảnh ngoài queue. reports/t07_ai_rework_report.md và out/poc_ai_corrected/rework_verification.json ghi kết quả xem lại, với independent_validation=false. Quyết định nhánh sửa là REWORK; bản máy giữ quyết định HOLD vì chưa có review đúng luồng và QA độc lập.

## 8 Tái lập và test

Windows: tạo .venv bằng py-3.13, pip install-r requirements.txt, chuẩn bị weights trước bằng python-m tools.prepare_t07_models. Khi demo, kiểm weights với --offline; các lệnh đầy đủ trong README và docs/T07_RUNBOOK.md. Standard requirements đã bao gồm dependency T07; kiểm pip dry-run không mạng đạt trên môi trường hiện có, chưa tuyên bố đã thử fresh venv trên mọi OS.

```powershell
.\\.venv\\Scripts\\python.exe -m tools.prepare_t07_models --offline
.\\.venv\\Scripts\\python.exe -m src.main --track t07 --output out/poc --run-name poc --offline --seed 42
.\\.venv\\Scripts\\python.exe -m tools.export_t07_ai_review
.\\.venv\\Scripts\\python.exe -m tools.apply_t07_ai_corrections
.\\.venv\\Scripts\\python.exe -m tools.verify_t07_ai_rework
.\\.venv\\Scripts\\python.exe -m pytest -q
```

Pytest thật: {validation['passed']} passed, {validation['skipped']} skipped, {validation['failed']} failed; {validation['t07_tests']} test T07. Skip do thiếu GT giảng viên. Model-quality metric không được suy ra từ test pass. Máy: Python {info['machine']['python']}, {info['machine']['platform']}, CPU {info['machine']['processor']}.

Commit code/evidence khi dựng report: {commit}. Snapshot đầu được giữ dù code review/audit được bảo trì; detector/config/model hash vẫn được kiểm không đổi.

## 9 Giới hạn và quyết định nộp

Chưa đủ cơ sở GO hoặc kết luận đạt metric. Các giới hạn gồm domain synthetic, heuristic không phải model head, AI tự kiểm bản sửa, thiếu GT độc lập, license weights chưa rõ và không có benchmark phần cứng cô lập. Không tuyên bố đạt GDPR/luật Việt Nam hoặc ẩn danh hóa hoàn toàn.

Nội dung5 slide và kịch bản demo trong demo/showcase.md; chưa xuất deck PPTX vì runtime tạo deck không có trong phiên. Không tạo tag final-24h để giả trạng thái hoàn tất; tag demo/final chỉ ghi khi thực sự freeze đúng nội dung sẽ nộp.

## 10 Đóng góp và Git

| Học viên | Git identity cấu hình hiện có | Vai trò | Evidence |
|---|---|---|---|
| Nguyễn Văn Hùng, 202603097 | {name} / {email} | Owner/integration, bài nộp được Codex hỗ trợ code và AI visual review | {commit}; reports/team_contributions.csv |

Không tự thêm thành viên hoặc claim người thật đã review ảnh. Đóng góp source có AI hỗ trợ được ghi minh bạch; tên reviewer trong CSV phản ánh đúng actor đã thực hiện.

## 11 References

- docs/tracks/T07.md và README: output contract, metric chuẩn, offline/GT guards, rubric.
- https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet — chọn detector mặt CPU và nguồn/license weights; xem2026-11-14.
- https://github.com/ankandrew/open-image-models — ONNX plate inference CPU, class map plate; xem2026-11-14.
- https://github.com/ankandrew/open-image-models/issues/14 — trạng thái license weights cần ghi rõ; xem2026-11-14.

## 12 Checklist hồ sơ

- Có run offline, source/license inventory, manifest, seed và test.
- Có baseline/PoC/blur-all và bằng chứng failure cases; metric GT còn thiếu.
- Có review/sửa AI với provenance và audit; không gán thành human evidence.
- Report và nội dung demo đã điền; deck, đánh giá GT và freeze/tag cuối chưa hoàn tất.
"""
    (ROOT / "reports/group_report.md").write_text(report, encoding="utf-8")
    contributions = pd.DataFrame([{"member": "Nguyễn Văn Hùng (202603097)", "git_author_name": name,
                                  "git_email": email, "role": "Owner/integration, Codex-assisted implementation",
                                  "modules_files": "src/t07_*;tests/test_t07_pipeline.py;tools/*t07*;reports/;configs/lab.yaml",
                                  "commit_shas": commit, "evidence_note": "Code and AI visual review assisted by Codex; no human image-review action claimed",
                                  "presenter": "not recorded"}])
    contributions.to_csv(ROOT / "reports/team_contributions.csv", index=False)
    print(f"Built group_report.md and team_contributions.csv from artifacts; code evidence {commit}")


if __name__ == "__main__":
    main()
