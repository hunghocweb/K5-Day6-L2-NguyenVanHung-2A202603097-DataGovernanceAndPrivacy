# T07 — Thử nghiệm thuật toán toán học trên ảnh 2D

## 1. Kết quả đối chiếu mục tiêu

**Chưa đạt yêu cầu tổng thể.** Bản đã khóa đạt mục tiêu độ phủ dev ≥95% và val ≥90%, nhưng che thừa trên val là 50,22%, vượt giới hạn 30%. Trạng thái **REWORK_OVER_REDACTION**; chỉ cung cấp chế độ thử trên web, giữ nguyên cấu hình pipeline mặc định.

## 2. Dữ liệu và protocol

Giữ split gốc 36 dev/24 eval và seed 42. Chạy từ pixel đủ 60 ảnh; chỉ 9 ảnh dev/49 vùng và 6 ảnh eval/31 vùng có bbox người dùng đã duyệt để tính chất lượng. Không coi 45 ảnh còn lại là negative. Những nhãn này nằm trong đúng queue 15 ảnh đã cố định, không tăng ngân sách review.

Đây là nhãn tham chiếu nội bộ lấy từ vùng che đã duyệt, có padding. Không thay GT giảng viên. Dữ liệu từng được xem/review ở vòng trước nên val không phải holdout độc lập hoàn toàn. Trong vòng toán học này chỉ chỉnh trên dev; khóa nguồn/phụ thuộc/config/environment trước khi đọc val, chạy val một lần và giữ nguyên thuật toán sau kết quả.

## 3. Thuật toán

1. Ước lượng Gaussian blur bằng tỷ lệ năng lượng gradient/Laplacian. Tạo các view Wiener gain 0,8/1,0/1,2, regularization 0,003; giữ ảnh gốc cho kiểm tra và xuất kết quả.
2. Nhánh mặt thô dùng mask màu và ellipse/nét mặt v3 trên ảnh gốc và các view phục hồi. Hợp nhất giữ độ phủ; xác nhận lại bằng chứng mắt–miệng trên pixel gốc để loại nét giả do ringing.
3. Nhánh mặt tinh tìm contour Canny/MSER trong các thành phần màu cục bộ, fit ellipse bằng AMS/moment. Cung đầu bị che được hoàn thiện bằng RANSAC có lọc độ cong và giới hạn ngoại suy. Chuẩn hóa ellipse, kiểm tra nét tối trên cả luma/chroma và so ellipse trống với ellipse có nét mặt. Mặt nhỏ v4 có khung tối đa 20 px bổ sung các trường hợp nhỏ.
4. Nhánh biển số tìm thân xe màu tương đối đồng nhất có bằng chứng cửa kính. Chiếu đối nghịch màu R−G/B−G để tìm panel trắng/đen gần trung tính ngay cả khi blur hòa chữ vào nền. Nối các mảnh cùng hàng trong thân xe; ràng buộc tỷ lệ và vị trí để hạn chế dính đèn.
5. Margin biển chọn trên dev: ngang tối thiểu 8 px, dọc tối thiểu 4 px, có sàn theo chiều cao nét. Không mở rộng theo biển thô. Mặt thô nhỏ thêm 1 px; hợp nhất cùng lớp giữ độ phủ. Blur sigma 12 trên ảnh gốc trong các bbox cuối.

Suy luận trong `src/t07_math2d.py` chỉ nhận mảng pixel. Không đọc nhãn, tên ảnh, manifest hay dự đoán lưu sẵn. Đây là cải thiện thuật toán 2D; không huấn luyện lại YuNet/YOLO.

## 4. Kết quả và cách tính

Một vùng thành công khi hợp mask che phủ ≥90% vùng tham chiếu. Success=1−PII Miss Rate. Che thừa=pixel che ngoài hợp GT/tổng pixel che. Dùng hàm metric của repo; CI bootstrap theo ảnh 1000 lần, seed 42. Đây không phải mAP hoặc confidence detector.

| Tập | Ảnh chạy / ảnh có nhãn | Vùng đủ phủ | Miss | Che thừa | Gate ≤30% |
|---|---:|---:|---:|---:|---|
| dev | 36 / 9 | 47/49 (95.92%) | 4.08% | 28.55% | PASS |
| val (eval) | 24 / 6 | 30/31 (96.77%) | 3.23% | 50.22% | FAIL |

| Tập | Mặt đủ phủ | Biển số đủ phủ | CI 95% success | CI 95% che thừa |
|---|---:|---:|---|---|
| dev | 26/26 | 21/23 | 90.38%–100.00% | 23.64%–35.06% |
| eval | 16/16 | 14/15 | 90.62%–100.00% | 40.69%–59.04% |

Runtime chạy 36 dev: 367.73 s; chạy 24 eval: 245.04 s. Đây là thời gian inference và xuất bằng chứng máy, không bao gồm review người.

## 5. Kiểm chứng

- Toàn repo trước nối web:124 passed,39 skipped,0 failed. Cả 39 skip do thiếu overlay GT giảng viên.
- Sau nối web:12 test web/cascade passed. Bộ ảnh tự tạo kiểm tra đầu có nét/đầu trống, biển trên xe, blur, thu nhỏ, JPEG và đầu bị che; không dùng crop GT làm template.
- Chrome thực: desktop 13 và mobile 12 kiểm tra pass: upload/worker, đủ 6 bước, tải ZIP, mở lại phiên, mode/margin đúng và không tràn ngang. Ảnh smoke làpii_40 thuộc dev; không phải phép đo độ chính xác trên ảnh mới.
- Hash nguồn/phụ thuộc/config của dev và val trùng bản freeze; journal human gốc không đổi. Audit mới là sự kiện máy, không ghi thành quyết định duyệt của người dùng.

## 6. Tái lập và chạy thử

Bấm `MO_XEM_TUNG_BUOC_T07.cmd`, chọn **2D toán học · ellipse và biển số mờ**, tải ảnh và chạy. Margin bổ sung tự về 0% vì cascade đã có margin nội bộ. Với ảnh thật, chọn detector đa tỷ lệ. Nếu server đang dùng mã cũ, dừng server local rồi mở lại launcher.

```powershell
.\.venv\Scripts\python.exe -m tools.evaluate_t07_math2d --split dev
.\.venv\Scripts\python.exe -m tools.build_t07_math2d_report
.\.venv\Scripts\python.exe -m tools.smoke_t07_walkthrough --mode math2d
```

Không chạy lại val: marker `reports/t07_math2d_validation_once.json` đã ghi complete và evaluator chặn lần đo sau. Kết quả đã lưu đủ để kiểm tra lại phép tính.

## 7. Giới hạn và công việc còn lại

Độ phủ cao chưa đủ để đạt lab: val còn vượt ngưỡng che thừa. Hợp nhất bảo thủ và margin được chọn trên 9 ảnh dev là giới hạn cần đánh giá thêm; số liệu hiện tại chưa chứng minh tổng quát sang ảnh thật hoặc mọi kiểu vẽ/blur. Không chọn lại threshold hay sửa hình học dựa trên val này.

Vòng tiếp theo cần mở rộng nhãn trên phần dev chưa có nhãn, thêm ca âm/ảnh mới có quyền dùng và chuẩn bị holdout mới trước khi tune tiếp. Chưa đổi quyết định lab thành GO và chưa ghi người dùng đã duyệt đầu ra mới.

## 8. Bằng chứng

- `reports/t07_math2d_experiment.json`: metrics/CI, status và kiểm tra nguồn.
- `reports/t07_math2d_freeze.json`: mã/config/environment khóa trước val.
- `reports/t07_math2d_validation_once.json`: dấu lần đo val duy nhất.
- `reports/evidence/math2d/`: metrics, predictions, coverage, trace, audit máy, JUnit và QA web.
- `reports/evidence/math2d/source/`: snapshot mã được đo.
- `reports/evidence/reviewed_reference/`: nguồn nhãn duyệt và provenance.
