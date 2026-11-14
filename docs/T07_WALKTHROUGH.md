# T07: tải ảnh và xem từng bước xử lý

Từ thư mục gốc repo, chạy `python -m tools.open_t07_web --port 8768 --page /walkthrough`,
hoặc mở http://127.0.0.1:8768/walkthrough sau khi server chạy. Tạo run PoC trước
theo runbook. Không mở trực tiếp file HTML; các launcher Windows local chỉ là tiện ích mở trang.

Trang có một luồng tự động, version `auto-cascade-1`:

- Ảnh 2D phẳng: bản elip + phục hồi mờ `math2d-dev-1` đã đóng băng.
- Ảnh 2D nhiều chi tiết hoặc bản gốc không có phát hiện: bộ 2D chi tiết dự phòng.
- Ảnh thật: YuNet + YOLO. Ảnh chưa rõ loại giữ cơ chế kiểm tra cả hai tuyến.

Không chọn thuật toán thủ công và không có hiệu chỉnh bbox bằng hệ số học.
Bản 2D gốc giữ **margin có sẵn của cấu hình gốc**; đây không phải bản đã bỏ margin.
Thanh trượt chỉ nới bổ sung và mặc định **0%**.

1. Chọn hoặc kéo thả ảnh JPG, PNG, WebP, tối đa 15 MB / 24 megapixel.
2. Chọn **Nới vùng che bổ sung** từ 0–25%. Ví dụ 6% nới thêm mỗi phía theo 6%
   chiều tương ứng của bbox; 0% giữ vùng che của nhánh được chọn.
3. Bấm **Chạy xử lý ảnh**; xem tuyến đã chọn và từng bước bên trái.
4. Tải PNG hoặc ZIP gồm ảnh, `redactions.csv`, `report.json`.

| Bước | Ảnh | Công việc |
|---|---|---|
| 1. Ảnh đầu vào | source.png | Xoay EXIF, nền trắng cho alpha, thu nhỏ giữ tỷ lệ tới cạnh dài 1280 px |
| 2. Nhận biết loại ảnh | tiles.png | Router đo đặc trưng pixel; ảnh ở bước này chưa bị thay đổi |
| 3. Phát hiện theo nhánh | detections.png | Thật: YuNet + YOLO; 2D: bản gốc hoặc bộ chi tiết dự phòng |
| 4. Gộp vùng trùng | merged.png | Vùng của nhánh được chọn; bản gốc đã có mức nới sẵn |
| 5. Nới vùng che | padding.png | Áp phần trăm nới bổ sung của thanh trượt một lần |
| 6. Hoàn tất | result.png | Gaussian blur sigma 12 trên ảnh đầu vào làm việc |

Router còn thử nghiệm; khi không chắc loại ảnh, chạy hai nhánh và đánh dấu cần
xem lại. Model ảnh thật chỉ được nạp khi tuyến cần tới nó. Nhánh 2D dò trên ảnh
gốc và view phục hồi blur, xác nhận bằng nét mặt/nét chữ; không đọc nhãn, tên
ảnh hay kết quả đã duyệt. Bản 2D gốc giữ cấu hình đã đóng băng để tái lập;
nhánh 2D chi tiết dùng biên native và nới bổ sung do người dùng chọn.
Mảnh phát hiện chạm mép crop ảnh thật được giữ nếu không có khung đầy đủ thay thế,
và được đánh dấu cần kiểm tra.

Luồng mới dùng [kế hoạch cascade](T07_CASCADE_PLAN.md) và [bộ 2D chi tiết](T07_DETAILED_2D.md).
Cổng dùng mức tập trung bảng màu và mật độ cạnh sau khi bỏ viền đen, không dùng
tên file hoặc nhãn. Đây là cổng phù hợp thực nghiệm, **không đo được accuracy của
ảnh mới**. Khi bản gốc phù hợp nhưng không có phát hiện, chạy nhánh dự phòng;
đầu ra rỗng vẫn không chứng minh ảnh không có PII.

`report.json` ghi `routing`, `cascade.selected`, lý do chọn, đặc trưng cổng,
`detector_padding_applied`, `padding_policy`, `user_padding_applications=1`,
`bbox_calibration_applied=false`, `human_review=false`, `quality_metrics=null`.
Với bản gốc, `padding_applications=null` để không gọi sai margin nội bộ là một
lần nới duy nhất. “Hoàn tất” chỉ nghĩa là xử lý xong; cần xem vùng lộ và vùng che nhầm.

Đã tái lập từ pixel trên 9 ảnh dev/49 vùng có nhãn: **47/49 (95,92%)** đủ phủ,
che thừa **28,55%**, nới bổ sung 0%. Cổng giữ bản gốc cho 36/36 dev; 27 ảnh chưa
có nhãn không được tính điểm chất lượng. Không dùng val để chỉnh trong vòng này.
Mục tiêu che thừa dưới 10% chưa đạt. Thêm 6% ngoài margin gốc làm che thừa tăng
lên 44,79% trên cùng dev, nên không dùng làm mặc định.

Các phiên cũ vẫn mở được bằng URL đã lưu và có nhãn **phiên cũ**. Điểm số trong
[báo cáo precision cũ](../reports/t07_precision_dev_experiment.md) thuộc cấu hình
đã đóng băng có hiệu chỉnh, không được dùng làm điểm của luồng mới.
Các quyết định duyệt cũ không tự áp vào kết quả mới.

Chạy cùng dispatcher qua CLI, chỉ tập dev:

```powershell
.\.venv\Scripts\python.exe -m src.main --config configs/lab_auto_dev.yaml --offline --seed 42
```

Output riêng `out/poc_auto_dev/` có ảnh, dự đoán, trace tuyến và queue 15 ảnh.
Profile nộp cũ giữ nguyên để tái lập; không thay đổi journal duyệt.

Xử lý offline trên máy, không tải weights. Phiên upload lưu trong
`out/walkthrough/<job>/`, có ảnh chưa che nên giữ local. Các phiên không tự xóa.
Ảnh xuất theo kích thước làm việc; nếu đã thu nhỏ thì không phải độ phân giải gốc.

Kiểm tra chức năng:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_t07_auto.py tests/test_t07_walkthrough.py tests/test_t07_multiscale.py -q -o addopts=
.\.venv\Scripts\python.exe -m tools.smoke_t07_walkthrough
```

Smoke test chạy thật trên ảnh dev `pii_40`, kiểm tra 6 bước, một luồng tự động,
tuyến 2D, margin gốc được khai báo, nới bổ sung một lần, ZIP, desktop/mobile và mở lại phiên.
Bằng chứng tại `out/t07_walkthrough_qa_auto/`. Đây là kiểm tra chức năng,
không đo độ chính xác trên ảnh mới hay tập đánh giá.
