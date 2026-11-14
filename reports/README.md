# reports/

Bắt đầu từ [group_report.md](group_report.md), trả lời đúng mục00–11 của template Day06. [Checklist nộp](submission_checklist.md) phân biệt phần đã có, chưa đo và chưa đưa lên GitHub; [submission_files.txt](submission_files.txt) là danh sách đường dẫn cần đưa lên sau khi chủ repo duyệt.

| File | Nội dung / cách tạo |
|---|---|
| group_report.md | Bài báo cáo chính; python -m tools.build_t07_report đọc bằng chứng đã gom |
| metric_contract.md | Protocol/metric đã khai báo; snapshot freeze gốc được giữ |
| team_contributions.csv | Người/role/commit thực tế; không tự thêm thành viên hoặc claim đã showcase |
| review_queue.csv | Đúng15ảnh queue, rank/score, reviewer/decision/bbox từ journal web |
| decision_log.csv | Quyết định hiện tại REWORK, reason_code và owner; không tự GO khi queue đủ |
| redactions_review15.csv | Bbox cuối sau đúng budget15; bản phục vụ evaluator nằm tại evidence/review15/redactions.csv kèm seed/run metadata |
| t07_run_summary.json | Số vùng/runtime baseline/PoC/blur-all; python -m tools.summarize_t07; không phải điểm chất lượng |
| t07_current_validation.json / evidence/pytest.xml | Kết quả kiểm thử mới có bằng chứng máy sinh; các t07_*validation cũ giữ lịch sử riêng |
| t07_submission_audit.json | Đối chiếu remote/local,review15,gate,tests,freeze và hashZIP; python -m tools.prepare_t07_submission |
| t07_models.json | URL/version/hash/license của weights |
| evidence/ | Predictions/run_info nguồn, audit nguyên bản,review15manifest,3review ngoài queue, JUnit và danh mục tài liệu remote |
| t07_ai_* | Lịch sử AI hỗ trợ review/sửa; không thay thế quyết định người đã lưu sau đó |

## File metric chưa có

Những file dưới đây cần GT hợp lệ. Không tạo file rỗng, số0 hoặc bảng điểm giả chỉ để đủ tên file:

```powershell
python -m src.evaluate --track t07 --pred reports/evidence/baseline/redactions.csv --split eval --run-name baseline --out reports/metrics_baseline.json
python -m src.evaluate --track t07 --pred reports/evidence/poc/redactions.csv --split eval --run-name poc_machine --out reports/metrics_poc_machine.json
python -m src.evaluate --track t07 --pred reports/evidence/review15/redactions.csv --split eval --run-name poc --out reports/metrics_poc.json --slices-out reports/slices_poc.csv
python -m src.evaluate --track t07 --pred reports/evidence/blur_all/redactions.csv --split eval --run-name blur_all --out reports/metrics_blur_all.json
python -m src.compare reports/metrics_baseline.json reports/metrics_poc.json --out reports/table1_baseline_vs_poc.md
```

Các file metrics/table phải do script sinh; giảng viên sẽ chạy lại. Evaluator chuẩn có type/condition; size slices, Recall@B và CI còn cần script phụ khi có nhãn. Bản review15 là export lại quyết định nên runtime end-to-end sau review chưa đo, không lấy thời gian export thay thời gian inference.

`out/` vẫn gitignore. Bằng chứng cần người chấm đọc đã gom vào reports/evidence, demo/t07_samples và ZIP synthetic trong submissions. Không đưa toàn bộ out, model weights, dữ liệu thật hoặc gt lên repo.
