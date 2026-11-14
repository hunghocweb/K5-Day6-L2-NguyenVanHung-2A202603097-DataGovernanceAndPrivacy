# T07 — Cải thiện thuật toán 2D: giảm che thừa dưới 10% trên dev

## 1. Kết quả đối chiếu mục tiêu

Bản `precision-dev-1` đã chạy trực tiếp từ pixel đủ **36 ảnh dev**. Trên **9 ảnh/49 vùng có nhãn đã duyệt**, **47/49 vùng (95,92%)** đủ phủ và che thừa **9.35%**, đạt mục tiêu dev ≥95% và <10%. Trạng thái **DEV_TARGET_PASS_REVIEW_REQUIRED**; vẫn còn hai vùng chưa đạt coverage 90%, chưa tự chuyển quyết định lab thành GO.

| Cấu hình | Vùng đủ phủ | Mặt đủ phủ | Biển đủ phủ | Che thừa |
|---|---:|---:|---:|---:|
| `math2d-dev-1` | 47/49 | 26/26 | 21/23 | 28.55% |
| precision-dev-1 | 47/49 | 25/26 | 22/23 | 9.35% |

Che thừa giảm **19.20 điểm phần trăm**. Tổng số vùng đủ phủ giữ nguyên; có một mặt giảm coverage và một biển được cải thiện, nên cần xem theo từng loại thay vì chỉ nhìn số tổng.

## 2. Dữ liệu và phép đo

Chỉ dùng ảnh, dự đoán và nhãn dev. Bbox tham chiếu lấy từ quyết định người dùng đã duyệt trong queue 15 cố định; không thêm lượt duyệt, không coi 27 ảnh dev chưa có nhãn là negative, không thay nhãn hoặc công thức metric. Đây là nhãn nội bộ có padding, không phải GT giảng viên.

Một vùng thành công nếu hợp mask che phủ ít nhất 90% vùng tham chiếu. Che thừa=pixel che ngoài hợp GT/tổng pixel che. Bản cuối dùng metric repo; phép nén tọa độ dùng khi tìm tham số đã được kiểm tra cho kết quả đúng bằng raster pixel, gồm clipping, tọa độ lẻ và vùng chồng nhau.

Số đếm: 4408 pixel ngoài GT / 47120 pixel che. CI 95% theo bootstrap ảnh 1000 lần, seed 42: success 90.38%–100.00%; che thừa 7.53%–11.87%. CI che thừa còn vượt 10% ở cận trên; điểm trên 9 ảnh không phải bảo đảm thống kê cho mọi ảnh mới.

## 3. Thuật toán được chọn

1. Dùng cascade toán học trước đó tạo đề xuất, giữ ảnh gốc để kiểm chứng. Các view Wiener chỉ hỗ trợ phát hiện.
2. Khớp lại contour ellipse từ mức trộn màu trên ảnh gốc; margin mặt phụ thuộc blur. Mặt nhỏ loại padding dư, bắt buộc mô hình nét mắt và miệng cùng có hệ số dương, đủ giải thích phần tín hiệu còn lại sau khi loại nền ellipse trống.
3. Hiệu chỉnh biên mặt bằng **một ma trận hệ số chung**: z=(f−mean)/scale, delta=z×beta; delta điều chỉnh tâm và kích thước theo chiều cao khung. Sáu đặc trưng gồm bias, nghịch đảo chiều cao, blur/chiều cao, tỷ lệ khung, luminance và độ tương phản cục bộ. Ridge alpha 30 được fit trên 22 vùng/8 ảnhdev; không có ảnh mẫu, image_id, tọa độ GT hoặc bảng tra từng ảnh trong model.
4. Biển số lấy nền gần trung tính trong thân xe. Cửa kính phải có hai cạnh nằm thực sự phía trên thân, cách halo blur của chính ứng viên. Nét chữ dùng cực tiểu dọc hai phía ở nhiều khoảng cách: S=min(G(y−d)−G(y),G(y+d)−G(y)); tích lũy theo cột, nối dải chữ vào nền biển và loại đèn có lõi sáng bão hòa. Dải chữ dài đi qua nền biển được giữ lại ngay cả khi blur nối chữ thành một khối.
5. Margin biển không cộng hai lần và được tách bốn phía: ảnh sắc dùng trái 2/phải 5/trên 1/dưới 1 px; ảnh blur dùng trái 0/phải 3/trên 3/dưới 2 px. Khi biển rất nhỏ bị che chỉ còn một mảnh, che bảo thủ vùng xe nhỏ có giới hạn và đánh dấu cần review.
6. Blur sigma 12 trên ảnh gốc trong bbox cuối. Xuất tọa độ, trace máy và ảnh; không tạo quyết định human.

Bản chọn không dùng bộ hồi quy biên biển số: thử nghiệm đó có điểm fit tốt nhưng không ổn định khi bỏ từng ảnh. Cũng không dùng voting tọa độ làm quyết định cuối; biên lớn từ bước đề xuất được thay bằng biên đã khớp và hiệu chỉnh. Không huấn luyện lại YuNet/YOLO.

## 4. Kiểm tra độ ổn định và giới hạn

Bỏ lần lượt từng ảnh khỏi phần học hệ số mặt, giữ nhánh biển và các tham số đã chọn: **46/49 (93,88%)**, che thừa **9.64%**. Đây chỉ là chẩn đoán mô-đun mặt trên dev; không phải kiểm định độc lập cả pipeline và chưa đạt mốc 95% về độ phủ. Vì vậy **chưa chứng minh đã tránh được overfit**.

- Hai vùng còn thiếu coverage 90%: mặt trong `pii_10` và biển trong `pii_38`. Có crop nguồn/kết quả/tham chiếu trong evidence để xem trực tiếp.
- Mẫu nhãn nhỏ, lấy từ queue rủi ro và có padding do người duyệt; kết quả không đại diện toàn bộ 36 ảnh hoặc ảnh thật.
- Các ca hình tự tạo khác chỉ kiểm coverage vật thể, blur/scale/JPEG và loại đầu trống; không dùng chúng để tuyên bố che thừa<10% trên dữ liệu mới.
- Cần bổ sung nhãn dev có nguồn độc lập và kiểm tra thêm trước khi dùng làm kết luận tổng quát. Chế độ web vẫn là bản thử.

## 5. Kiểm thử và tái lập

Toàn repo: **145 passed, 39 skipped, 0 failed**. Các skip do thiếu overlay GT giảng viên. Hai mươi mốt kiểm thử mới gồm hình học, bằng chứng mặt nhỏ/cửa kính, không nối nền tối, metric nén đúng pixel, ba ca toàn cascade có blur/scale/JPEG và tích hợp pipeline không khởi tạo detector ảnh thật, không nới bbox thêm hoặc tạo sự kiện human.

Chrome thực: desktop 13 và mobile 12 kiểm tra pass; upload, đủ 6 bước, ZIP và mở lại đúng mode/margin. Đã xem ảnh QA; ảnh smoke thuộc dev.

```powershell
.\.venv\Scripts\python.exe -m tools.evaluate_t07_precision_dev
.\.venv\Scripts\python.exe -m tools.render_t07_precision_dev
.\.venv\Scripts\python.exe -m tools.build_t07_precision_dev_report
.\.venv\Scripts\python.exe -m tools.smoke_t07_walkthrough --mode precision
```

Profile one-command riêng cho dev đã nối `synthetic_algorithm: precision` vào pipeline T07. Lệnh dưới tạo output, queue và quyết định theo contract; không đổi profile mặc định hoặc tái gán quyết định duyệt cũ cho snapshot mới:

```powershell
.\.venv\Scripts\python.exe -m src.main --config configs/lab_precision_dev.yaml --offline --seed 42
```

Runtime chạy 36 dev của lần này: 312.56 s. Các kiểm tra khác chạy đồng thời nên chưa dùng để kết luận tốc độ tăng/giảm. Freeze gồm nguồn/config/environment và hash nhãn dev; journal human gốc giữ nguyên.

## 6. Chạy thử và bằng chứng

Bấm `MO_PRECISION_T07.cmd` để mở trang cổng 8767, chọn **2D precision · giảm che dư**. Margin bổ sung tự về 0% để giữ đúng cấu hình đã đo; nới thêm sẽ thay kết quả. Các bước hiển thị đề xuất → khớp biên/nét chữ → hiệu chỉnh → blur.

- `reports/t07_precision_dev_experiment.json`: metrics, so sánh dev và chẩn đoán bỏ từng ảnh.
- `reports/t07_precision_dev_freeze.json`: snapshot khóa nguồn/config.
- `reports/evidence/precision_dev/`: predictions, reference_dev, coverage, trace, audit máy, JUnit, ảnh kết quả và QA web.
- `configs/t07_precision.json`: một bộ hệ số/tham số chung, không chứa bbox theo từng ảnh.
- Thay đổi vẫn local; chưa thay cấu hình mặc định hoặc ghi người dùng đã duyệt đầu ra mới.
