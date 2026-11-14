# T07 Runbook G07

Owner: Nguyễn Văn Hùng, MSSV 202603097. Việc ghi owner không thay thế thao tác review thật.

## Chuẩn bị và dữ liệu

Đọc `docs/tracks/T07.md`. Dữ liệu synthetic có 36 dev, 24 eval. Chỉ dùng dev để thiết kế/tune. Ground truth do giảng viên giữ; không đọc `gt/` trong pipeline.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-t07.txt
.\.venv\Scripts\python.exe -m tools.prepare_t07_models
.\.venv\Scripts\python.exe -m tools.inspect_t07
```

Mở `out/t07_inspection/dev_contact_sheet.html`. Trang này chỉ để xem input, không phải bằng chứng người đã review prediction.

Weights cần có ở đường dẫn trong `reports/t07_models.json`. Tải trước từ URL `source`, đối chiếu SHA256; không dùng tên model trong factory để tự download khi chạy. Kiểm license trong inventory, đặc biệt tình trạng chưa rõ của weights YOLO. Không commit weights.

## Chạy baseline và PoC offline

```powershell
.\.venv\Scripts\python.exe -m src.main --config configs/lab.yaml --track t07 --pipeline src.t07_baseline --output out/baseline --run-name baseline --offline --seed 42
.\.venv\Scripts\python.exe -m src.main --config configs/lab.yaml --track t07 --pipeline src.t07_blur_all --output out/blur_all --run-name blur_all --offline --seed 42
.\.venv\Scripts\python.exe -m src.main --config configs/lab.yaml --track t07 --output out/poc --run-name poc --offline --seed 42
```

Baseline: YuNet + YOLO confidence mặc định, blur sigma 12, không sửa bằng người. PoC bổ sung rule phát hiện đầu có dấu hiệu hai mắt và biển số hình học trên ảnh synthetic; đây không phải model head pretrained. Không tìm được mắt vẫn là bất định cần review. PoC padding 3%, risk queue B=15. Không dùng kết quả eval để sửa các tham số này.

## Vòng review người thật

1. Mở `out/poc/contact_sheet.html` offline. Mỗi ảnh có original và ảnh máy đã che. Xem cả vùng bị bỏ sót, đầu quay lưng và vùng che quá rộng.
2. Nhập tên reviewer thực tế. `approve` giữ nguyên vùng đang hiển thị; ban đầu đó là vùng máy, sau sửa là bản xuất hiện tại. Nếu thấy lỗi nhưng không biết tọa độ, chọn `reject`, ghi rõ lý do và để trống override: đây là yêu cầu làm lại, không phải approve, không tự xóa bản sửa trước. `edit` cần toàn bộ danh sách bbox cuối trong ô override và ghi lý do. `skip` hoặc `escalate` giữ ảnh chưa được xử lý xong.
3. Bbox dùng pixel ảnh gốc 640×360: `x,y` góc trái trên; `w,h` chiều rộng/cao; `type` là `face` hoặc `plate`. Tham khảo `out/poc/review_box_examples.json`. Không dùng tọa độ thumbnail. Có thể chỉnh CSV trực tiếp nếu cần nhập danh sách dài.

Ví dụ thay toàn bộ vùng che một ảnh bằng hai vùng:

```json
[{"x":120,"y":60,"w":22,"h":26,"type":"face"},{"x":300,"y":250,"w":38,"h":12,"type":"plate"}]
```

Thêm vùng: giữ các vùng đúng và thêm object mới vào danh sách. Xóa một vùng: bỏ object tương ứng. Xóa tất cả vùng: nhập `[]` với `edit` hoặc `reject` và giải thích vì sao. Không tự xóa tất cả chỉ vì detector không thấy mặt.

`reject` để trống override + có lý do sẽ tạo `human_change_requests.csv` và quyết định REWORK. Máy giữ nguyên vùng hiện tại để không đoán tọa độ; Codex xử lý yêu cầu sửa riêng, không đổi tác giả/ý nghĩa quyết định của bạn. Ảnh sửa cần được người duyệt xem lại; checkpoint chưa hoàn tất chỉ vì đã ghi reject.

4. Bấm Export decisions CSV; chép file tải về vào `out/poc/review_queue_reviewed.csv`. Không tự điền approve hàng loạt. Giữ nguyên `item_id`, `reason` (snapshot) và timestamp do trang xuất.
5. Chạy lại đúng lệnh PoC. Hệ thống validate review, áp thay đổi từ ảnh gốc và ghi `audit_log.jsonl`, `review_diff.json`, `review_state.json`. Sau khi ảnh cuối đã xuất xong, `contact_sheet.html` được tạo lại từ `redacted/`; queue và thứ tự ưu tiên giữ nguyên. `review_box_examples.json` chứa vùng của bản xuất hiện tại.
6. Chạy lại cùng trạng thái review không tạo audit event lặp. Phục hồi file review A cũ sau B ghi thêm chuyển trạng thái B→A, giữ timestamp review gốc và đánh dấu `reapplied`; không giả một lần duyệt mới. Bỏ file/dòng review có thể phục hồi vùng máy, khi đó audit ghi `review:reset` với actor `pipeline`.
7. Kiểm ảnh trong `out/poc/redacted/` và tọa độ `redactions.csv`. CP07 chỉ DONE sau khi review người thật và kiểm audit; reviewer trong unit test không tính.

Pipeline hiện giữ HOLD nếu còn item chưa duyệt; ngay cả queue hoàn tất vẫn HOLD vì chưa có chứng cứ QA độc lập và phê duyệt release. Hoàn tất top-15 không chứng minh 60 ảnh không còn PII bị bỏ sót.

## Đánh giá và giới hạn

`detection_summary.json` chỉ báo số detection; không phải precision/recall. Chưa có GT thì không báo PII Miss Rate giả. Kiểm tra hạ tầng:

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts= -q
```

Khi giảng viên cung cấp GT mới chạy evaluator/compare theo README. Có thể gán nhãn tập dev nội bộ bằng người và ghi protocol riêng; không gọi nó là GT chính thức. Giữ metric chuẩn và báo blur-all cạnh PoC.
