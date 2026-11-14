# T07: thao tác review trên web local

**Mở bằng chuột:** bấm đúp `MO_WEB_T07.cmd` ở thư mục gốc repo. File này tự khởi động server nền nếu cần rồi mở trình duyệt. Bấm lại sẽ dùng server đang chạy, không tạo thêm server. Không cần mở các file HTML trong thư mục QA.

**Tải ảnh khác và xem từng bước:** bấm đúp `MO_XEM_TUNG_BUOC_T07.cmd`, hoặc bấm **Tải ảnh · xem từng bước** trên đầu trang. Trang riêng `/walkthrough` chạy detector local trên ảnh upload, hiện đủ 6 bước và cho tải PNG/CSV/báo cáo. Xem [hướng dẫn walkthrough](T07_WALKTHROUGH.md).

Chạy từ thư mục repo, sau khi đã có kết quả `out/poc`:

```powershell
.\.venv\Scripts\python.exe -m tools.t07_review_web
```

Mở http://127.0.0.1:8765. Server chỉ lắng nghe trên máy hiện tại. Không cần npm, không dùng CDN và không tải model khi review. Dừng bằng Ctrl+C trong terminal chạy server.

## Cách dùng

1. Chọn ảnh bên trái. Mặc định hiển thị queue 15 ảnh; có thể tìm mã ảnh hoặc lọc toàn bộ 60 ảnh, 36 dev, 24 eval.
2. Chuyển tab **Baseline**, **PoC tự động**, **Bản đang duyệt** để đối chiếu cùng ảnh gốc. Preview bản duyệt sử dụng cùng hàm blur OpenCV với pipeline.
3. Chọn **Vẽ vùng mặt** hoặc **Vẽ vùng biển số**, kéo trên ảnh gốc để tạo khung. **Chọn / di chuyển** cho phép kéo khung; kéo góc dưới phải để đổi kích thước. Có thể sửa chính xác tọa độ, loại vùng hoặc xóa trong cột bên phải. **Thêm vùng bằng tọa độ** hỗ trợ thao tác không dùng kéo chuột.
4. **Hoàn tác** phục hồi lần sửa vùng trước. **Khôi phục bản đã lưu** trở về vùng của lần lưu gần nhất cho ảnh đó; trước lần lưu đầu là bản nguồn của phiên web.
5. Nhập tên reviewer, chọn quyết định và ghi chú. Reject/escalate phải có lý do. Bấm **Lưu quyết định** hoặc **Lưu & ảnh tiếp**. Đóng trang/chuyển ảnh khi có sửa chưa lưu sẽ có cảnh báo.
6. Bấm **Xuất ảnh & nhật ký** để tải ZIP chứa đủ 60 ảnh theo trạng thái đã lưu; thay đổi chưa lưu không được xuất.

## Dữ liệu và lịch sử

- Phiên web bắt đầu từ vùng che trong `out/poc_ai_corrected/redactions.csv` nếu có, nếu không dùng `out/poc/redactions.csv`. Các quyết định cũ giữ ở thư mục cũ; tiến độ web chỉ tính hành động đã lưu trên web.
- Các file nằm trong `out/web_review`: `audit_log.jsonl` append-only; `review_queue_reviewed.csv`; `redactions.csv`; `decision.json`; PNG của ảnh đã lưu trong `redacted/`. Nút xuất dựng đủ 60 PNG vào `t07_review_export.zip` cùng các file trên và `session.json`.
- Reviewer nhập trên form được ghi cho hành động thực sự bấm Lưu. Journal giữ vùng trước/sau, thời gian lưu, nguồn ảnh và snapshot phiên. Khi khởi động lại, trạng thái được phục hồi từ journal.
- Hai tab sửa cùng một phiên bản: lần lưu cũ bị từ chối; tải lại để lấy trạng thái mới. Nếu nguồn predictions/queue thay đổi giữa các lần khởi động, server yêu cầu tạo phiên mới, ví dụ `--output out/web_review_v2`.
- CSV web mô tả phiên bắt đầu từ bản sửa, **không chép trực tiếp vào `out/poc`** để replay trên snapshot máy khác. File `redactions.csv` của ZIP có đúng các cột cần cho evaluator T07 và thêm `type`.
- Không tự đổi threshold/detector, không đọc GT. Duyệt đủ queue vẫn HOLD/INDEPENDENT_QA_PENDING; reject tạo REWORK. Chưa có metric độc lập nên không tự GO.

## Kiểm tra

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_t07_web.py -q -o addopts=
.\.venv\Scripts\python.exe -m tools.smoke_t07_web
```

Smoke test dùng Chrome headless, phiên QA riêng trong `out/t07_web_qa`, không ghi quyết định thử vào phiên của người dùng. Có ảnh chụp desktop/mobile và JSON kết quả. Sự kiện pointer được phát bằng JavaScript trong trình duyệt cho kiểm thử handler; đây không phải người thật thao tác hoặc kiểm thử trên thiết bị cảm ứng thật.
