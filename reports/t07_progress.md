# T07 G07 — tiến độ hiện tại

Thực hành từ 14:00 ngày14/11/2026 đến 12:00 ngày15/11/2026, giờ Việt Nam. Báo cáo chính: [group_report.md](group_report.md). Danh sách file nộp: [submission_checklist.md](submission_checklist.md).

| Phần việc | Trạng thái / bằng chứng |
|---|---|
| Dữ liệu / model / config | G07 T07,36dev/24eval, seed42; fixture verify, weights inventory/checksum; inference offline |
| Baseline / PoC / blur-all | Có ba run thành công; số vùng/runtime trong t07_run_summary.json; chưa có metric GT |
| Human review | Queue15 có14approve/1edit của Nguyễn Văn Hùng; journal giữ riêng nguồn sửa ban đầu và người duyệt |
| Bản xuất kiểm soát |355vùng,60PNG; chỉ sửa15ảnh,45ảnh ngoài queue giữ nguyên; reports/evidence/review15/results.zip |
| Quyết định |HOLD/INDEPENDENT_QA_PENDING; phạm vi queue15 đã duyệt |
| Web | Bàn duyệt và trang upload/xem6bước chạy local; không train weights mới |
| Kiểm thử |159passed,39skipped,0failed; t07_current_validation.json và evidence/cascade_dev/pytest_publish.xml |
| Hồ sơ | Report đúng mục00–11; contributions/queue/decision/audit/failure PNG đã gom; xem checklist |
| GitHub |Mã nguồn và hồ sơ đã push lên main; có tag demo-180m và final-24h |
| Demo |Có 5slide HTML/PDF và kịch bản; chưa trình bày |
| Đánh giá |Có kết quả nội bộ trên9dev/49vùng; chạy metric chính thức khi giảng viên cấp overlayGT |
