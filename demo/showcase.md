# T07 G07 Showcase

Nguyễn Văn Hùng · MSSV 202603097. Bộ 5 slide có [HTML](T07_G07_slides.html) và [PDF](T07_G07_slides.pdf), đồng bộ với kịch bản dưới đây. Chưa trình bày demo.

## Slide 1 Bài toán và dữ liệu

Che mặt và biển số trong ảnh, ưu tiên các trường hợp rủi ro và ghi rõ quyết định giữ lại hay làm lại.

- 60 ảnh synthetic CC0, 640×360; 36 dev, 24 eval.
- B=15 ảnh review, seed 42, inference CPU offline.
- Không dùng ảnh người thật hoặc biển số thật trong bản demo này.

## Slide 2 Pipeline và provenance

Load/verify manifest → local YuNet + YOLO plate → bổ sung rule synthetic → risk queue → review/override/audit → export PNG và CSV → HOLD/REWORK.

- Model có nguồn, phiên bản và SHA256.
- GT guard và offline guard giữ nguyên.
- Nguồn và license model được ghi trong inventory.

## Slide 3 Baseline và kết quả máy

Dùng bảng do `tools.summarize_t07` sinh trong `reports/t07_run_summary.json`.

- Baseline: detector mặc định, blur cố định, không correction.
- PoC: cùng pool và model confidence, thêm rule/risk queue.
- Blur-all: đối chứng, không coi là lời giải đạt guardrail.
- Số bbox và runtime là thống kê vận hành; không dùng chúng để suy ra recall hoặc over-redaction.

## Slide 4 Hai failure case và vòng sửa

- `pii_10` (dev): chữ biển số vượt mép bbox; bản sửa theo ảnh che thêm phần chữ đang lộ.
- `pii_40` (dev): ảnh mờ làm sót mặt, vùng kính xe bị nhận nhầm thành plate; bản sửa bổ sung đúng vùng và bỏ vùng che nhầm.
- Mở các ảnh nguồn, PoC và review15 đã gom trong `demo/t07_samples/` để đối chiếu.
- Nguyễn Văn Hùng đã lưu 14 approve và 1 edit cho queue15; lịch sử sửa có AI hỗ trợ giữ riêng trong audit. Review không phải bộ nhãn độc lập.

## Slide 5 Quyết định và giới hạn

- Bản sửa xử lý 15 ảnh; 45 ảnh ngoài queue không đổi, audit replay-safe.
- Detector không được tune theo ghi chú eval; sửa riêng theo ảnh được lưu riêng.
- Chưa có GT độc lập để đo PII Miss Rate và guardrail che thừa ≤0.30.
- Phạm vi nộp là queue15 đã duyệt; 45ảnh ngoài queue giữ bản PoC máy. Quyết định HOLD/INDEPENDENT_QA_PENDING.

## Kịch bản thao tác

Chạy từ thư mục repo bằng PowerShell:

```powershell
.\.venv\Scripts\python.exe -m tools.prepare_t07_models --offline
.\.venv\Scripts\python.exe -m src.main --track t07 --output out/poc --run-name poc --offline --seed 42
.\.venv\Scripts\python.exe -m pytest -q
```

Các ghi chú AI và bbox sửa chỉ áp cho fixture/snapshot đã xem. Không chạy lại exporter để gán ghi chú cũ cho pool ảnh khác.

Chạy `python -m tools.open_t07_web --port 8768 --page /walkthrough` để tải ảnh/xem6bước; tạo run PoC trước theo lệnh trên. Gói review15 nộp tại [reports/evidence/review15/results.zip](../reports/evidence/review15/results.zip). Chỉ giới thiệu kết quả từ artifact đã chạy; không bấm approve để minh họa như một quyết định thật.

Nhánh mở rộng: ảnh2D phẳng/elip/mờ dùng bản gốc, ảnh2D nhiều chi tiết dùng bộ toán học dự phòng, ảnh thật dùng YuNet+YOLO. Trên9dev có nhãn:47/49vùng đủ phủ, che thừa28,55%; đây là kiểm tra nội bộ, không phải điểm heldout chính thức. Lệnh CLI riêng: `python -m src.main --config configs/lab_auto_dev.yaml --offline --seed 42`.

## Câu hỏi debate

1. Vì sao che kín toàn ảnh không phải lời giải đạt yêu cầu? Đối chiếu guardrail che thừa.
2. Nếu nhìn thấy chữ ngoài bbox nhưng GT chỉ bao nền biển số thì xử lý thế nào? Giữ metric chính thức và báo thêm rủi ro thị giác, không sửa GT.
3. Vì sao 159 test pass vẫn chưa chứng minh độ chính xác model? Test kiểm contract và lỗi workflow; chất lượng cần nhãn độc lập.
4. Vì sao audit phải ghi thêm khi phục hồi A sau B? Vì trạng thái xuất thực tế đổi B→A dù quyết định A đã từng xuất hiện.
