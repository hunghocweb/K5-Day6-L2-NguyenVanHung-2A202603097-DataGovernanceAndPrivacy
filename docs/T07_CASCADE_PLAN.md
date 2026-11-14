# CP20 — Giữ 2D gốc, dự phòng 2D chi tiết, ảnh thật dùng model

**Đã hoàn tất tích hợp và kiểm tra chức năng; chưa đạt che thừa dưới 10%.**

Theo yêu cầu mới, không thay bộ 2D gốc bằng prototype unified đang thử.

1. Giữ nguyên mã `math2d-dev-1` và cấu hình freeze: nhánh chính cho hình vẽ phẳng,
   elip và ảnh mờ. Tái lập số dev từ pixel, không lấy dự đoán đã lưu làm đầu ra.
2. Cổng phù hợp chỉ đọc pixel, bỏ viền đen khi tính đặc trưng; đo mức tập trung bảng
   màu và độ dày cạnh ở cùng kích thước phân tích. Điểm này không phải accuracy.
3. Hình 2D quá nhiều chi tiết so với điều kiện của bản gốc, hoặc bản gốc không có
   phát hiện, sẽ dùng `t07_native_2d` (bản 2D chi tiết đã có). Không hợp các khung
   nhận nhầm của nhánh bị loại vào ảnh cuối. Không dùng YuNet/YOLO trong hai nhánh 2D.
4. Giữ router thật/2D hiện tại. Ảnh thật dùng YuNet + YOLO; ảnh chưa rõ loại giữ
   cơ chế kiểm tra cả hai và yêu cầu review. Không dùng tên, ID hoặc manifest để định tuyến.
5. Ghi nhánh được chọn, lý do dự phòng và đặc trưng cổng trong output/log. Web vẫn
   một luồng; hiện rõ nhánh và vùng che. Bản gốc có margin cố định sẵn: ghi đúng
   `detector_padding_applied=true`, thanh nới **bổ sung** mặc định 0%. Không ghi
   giả là bản gốc không có margin hoặc đã đạt che thừa dưới 10%.
6. Kiểm tra dispatch, dự phòng khi đầu ra rỗng, không nạp model ảnh thật cho 2D,
   ảnh nguồn/CSV/pixel khớp; đo dev trước, chưa dùng val trong vòng này. Kiểm tra
   lại ảnh hoạt hình người dùng đã đưa; ghi worklog và chỉ làm local.

Số lịch sử của bản gốc trên 9 dev/49 vùng đã duyệt: 47/49 = 95,92% đủ phủ,
che thừa 28,55%. Cần tái lập; không gọi đây là độ chính xác toàn bộ 36 dev hoặc
độ chính xác của cổng. Mục tiêu che thừa dưới 10% vẫn chưa đạt.

Prototype `unified-2d-candidate-5` đã loại khỏi mã nộp khi dọn repo: kết quả lịch sử
39/49 đủ phủ ở nới 6%, che thừa 15,76%, không đạt mục tiêu. Không gán điểm hoặc
chất lượng này cho nhánh chính được khôi phục. Worklog giữ kết luận thử nghiệm.

## Kết quả tái lập

- `reports/evidence/cascade_dev/metrics.json`: từ pixel, 9 dev có nhãn/49 vùng, không nới
  bổ sung: 47/49 đủ phủ (95,92%), che thừa 17.536/61.422 pixel = 28,55%.
  Thêm 6% cho 48/49 đủ phủ nhưng che thừa 44,79%, nên mặc định giữ 0%.
- `reports/evidence/cascade_dev/gate_all_dev.json`: 36/36 dev phù hợp nhánh gốc;
  không phải 36/36 phát hiện đúng, vì chỉ 9 ảnh có nhãn tham chiếu.
- Cổng phân tích cạnh dài tối đa 640 px; 24 màu lượng tử hóa bước 24 phải chiếm
  ít nhất 88% pixel và mật độ cạnh Canny(40,100) không quá 4% để dùng bản gốc.
  Ngưỡng thực nghiệm chọn từ dev và ảnh upload dùng phát triển, không xác thực độc lập.
- Ảnh hoạt hình dùng phát triển: cổng chọn `detailed_2d`, 4 mặt và 6 biển,
  runtime 42,716 giây, nới bổ sung 0%. Đã xem khung và ảnh che; pixel ngoài vùng
  che giữ nguyên. Không gọi một ảnh này là điểm chất lượng độc lập.
  Phiên web: http://127.0.0.1:8768/walkthrough?job=8e42c9beb3fc41119e25c13bbb602bb8.
- Full suite tại lúc tích hợp: **168 passed, 39 skipped, 0 failed**, JUnit
  `reports/evidence/cascade_dev/pytest_before_cleanup.xml`.
  Các skip thiếu overlay GT giảng viên. Chrome desktop 17 check/mobile 16 check
  qua; đã xem cả hai screenshot, bản nộp tại `reports/evidence/cascade_dev/`
  (`web_results.json`, `web_desktop.png`, `web_mobile.png`).
- Hash toàn bộ source trong freeze bản gốc khớp. Journal human vẫn
  `d60d1e1fff07581602c6d997852e13998fdde437b58d5c70386de111b8fb5604`.
  Không tạo quyết định human, không dùng val trong vòng này, không commit/push.

Sau khi dọn prototype, script dò thử và cache: **159 passed, 39 skipped, 0 failed**,
JUnit `reports/evidence/cascade_dev/pytest.xml`. Số test giảm vì bỏ các test của
prototype không dùng; các test của pipeline/ba nhánh đang chạy vẫn qua. Toàn bộ
14 manifest và checksum hai weights đều hợp lệ. README ở gốc giữ nguyên nội dung.
Danh sách file nộp hiện có: `reports/submission_files.txt`.

Tái lập:

```powershell
.\.venv\Scripts\python.exe -m tools.verify_t07_cascade_dev --reviewed-only
.\.venv\Scripts\python.exe -m tools.smoke_t07_walkthrough --mode auto
```
