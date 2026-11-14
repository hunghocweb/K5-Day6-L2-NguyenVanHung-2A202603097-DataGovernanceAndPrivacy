# Gói kết quả T07 G07

`T07_G07_review15_candidate.zip` chứa60 PNG đã che, `redactions.csv`,15 quyết định review, quyết định batch, manifest và journal nguồn. Đây là **candidate / REWORK**, chưa phải release đã đạt metric.

Chỉ các bbox của queue15 được áp;45 ảnh ngoài queue khớp PoC gốc theo byte PNG. Ba lượt web thêm được báo riêng trong `reports/evidence/additional_web_reviews.csv`; hai reject còn mở nên không tự GO. Journal đầy đủ giữ nguồn AI hỗ trợ sửa và người review đã lưu quyết định; việc đóng gói không tạo hành động review mới.

Ảnh đều từ fixture synthetic có sẵn, không có ảnh thật từ upload. Gói tạo bằng `python -m tools.prepare_t07_submission` sau khi có run/review local; báo cáo dựng bằng `python -m tools.build_t07_report`. Các script đóng gói không thay thế một lần chạy inference hay một lần review của người.
