# Báo cáo thực hành Day 06 — T07 Face/Plate Redaction

## 00 · Header

| Mục | Giá trị |
|---|---|
| Group ID / Challenge | **G07 / T07 — Che mặt, biển số và ưu tiên ảnh cần duyệt** |
| Thành viên và vai trò | **Nguyễn Văn Hùng — 2A202603097**: tích hợp, thuật toán, kiểm thử, review và report. |
| Git URL | [Repo G07](https://github.com/hunghocweb/K5-Day6-L2-NguyenVanHung-2A202603097-DataGovernanceAndPrivacy) |
| Tag `demo-180m` / `final-24h` | [`demo-180m`](https://github.com/hunghocweb/K5-Day6-L2-NguyenVanHung-2A202603097-DataGovernanceAndPrivacy/tree/demo-180m) / [`final-24h`](https://github.com/hunghocweb/K5-Day6-L2-NguyenVanHung-2A202603097-DataGovernanceAndPrivacy/tree/final-24h). |
| Thời gian thực hành (giờ Việt Nam) | 14:00 ngày 14/11/2026 đến 12:00 ngày 15/11/2026. |

## 01 · Problem statement

- **Ai chịu hậu quả khi sai?** Người trong ảnh có thể bị lộ mặt/biển số; người dùng dữ liệu mất tín hiệu khi che nhầm hoặc che quá rộng.
- **Điểm nghẽn:** mặt nhỏ/tối/mờ dễ bị bỏ sót; bbox biển số có thể chưa phủ hết ký tự; kính xe và đầu quay lưng có thể bị che nhầm.
- **Đầu vào → đầu ra:** `images.csv`, ảnh và manifest → `redactions.csv` (`image_id,x,y,w,h`), PNG đã che, queue 15, review, audit và quyết định. [Gói 60 ảnh và nhật ký review15](evidence/review15/results.zip).
- **Quyết định:** **HOLD / INDEPENDENT_QA_PENDING**, owner **Nguyễn Văn Hùng**; phạm vi review là queue 15 ảnh đã duyệt.
- **Phạm vi dữ liệu:** bài nộp sử dụng fixture synthetic; ảnh thật upload thử được giữ local.

## 02 · Data & provenance

| Split | N ảnh | Nguồn / link | License | Synthetic? |
|---|---:|---|---|---|
| dev | 36 | [Fixture T07](../fixtures/t07_redaction/images.csv) của starter | CC0-1.0 theo manifest | Có |
| eval | 24 | Cùng fixture; giữ nguyên cột split | CC0-1.0 theo manifest | Có |

- **Schema / class mapping:** ảnh 640×360, bbox pixel `x,y,w,h`; hai loại `face`, `plate`. Theo đề, đầu quay lưng là hard negative.
- **Seed / chia split:** seed 42, dùng split có sẵn; không đổi ảnh hoặc chia lại để tăng điểm.
- **Manifest:** đã kiểm SHA256; lệnh `python -m tools.verify_manifests`. Pipeline kiểm lại trước inference.
- **Ảnh thật / GT:** chưa dùng tập ảnh thật có nhãn. GT chính thức do giảng viên giữ; repo học viên không có `gt/`. Vùng che đã sửa chưa phải bộ nhãn vật thể độc lập.

## 03 · Research & design

- **References / lý do chọn:** YuNet chạy CPU offline; YOLOv9-t dùng cho biển số. Detector mặt nhận kém mặt hoạt hình nên PoC bổ sung rule màu/hình dạng và dấu hiệu mắt. Nguồn và ngày xem ở mục 11.
- **Data flow:** kiểm hash → detect mặt/biển số → rule synthetic, NMS, padding → tính risk → top-15 → reviewer sửa → xuất ảnh/CSV, audit, quyết định.
- **Risk score:** `3×không có mặt + 2×đầu mơ hồ + 1.5×ảnh tối + 0.5×vùng nhỏ + 0.25×vùng confidence thấp`; thứ tự ổn định, không đọc GT.
- **Offline / weights:** YuNet 2023mar và YOLOv9-t plate ONNX; URL, SHA256, phiên bản và license tại [inventory](t07_models.json). Thiếu weights/sai hash báo lỗi; inference không tự tải model.
- **Bản mở rộng hiện tại:** [auto-cascade-1](../docs/T07_CASCADE_PLAN.md) chọn tuyến từ pixel: 2D phẳng/elip/mờ dùng `math2d-dev-1`; 2D nhiều chi tiết hoặc bản gốc không có phát hiện dùng bộ 2D chi tiết; ảnh thật dùng YuNet + YOLO. Cổng đo mức phù hợp của ảnh, không biết accuracy của ảnh mới. Hai nhánh 2D chỉ dùng phép toán.
- **Cấu hình cascade:** thanh nới bổ sung mặc định 0%; profile `configs/lab_auto_dev.yaml` chạy dev, tách khỏi PoC gốc đã tạo review15.

## 04 · Baseline vs PoC

- **Baseline / version:** YuNet + YOLO mặc định, blur sigma 12, không correction; `src.t07_baseline`.
- **PoC / version:** cùng weights/confidence, thêm rule synthetic, padding 3%, risk queue và sửa vùng; `src.pipeline` → `src.t07_pipeline`. Các run gốc ghi commit `8f1e0db`; [config](../configs/lab.yaml), [freeze](t07_protocol_freeze.json).
- **Cùng điều kiện:** 60 ảnh, 36 dev / 24 eval, seed 42; evaluator dùng 24 eval. Baseline không dùng người sửa theo định nghĩa đề; PoC sau review dùng B=15. Báo riêng PoC máy để tách tác dụng thuật toán và thao tác người. Ba lượt web ngoài queue được giữ riêng, không áp vào bản review15.

**Bảng 1 — Metric chất lượng:** chưa có output `src.compare` do thiếu GT; các ô sau chỉ ghi trạng thái, không phải điểm đã đo.

| Metric | Baseline | PoC máy | Sau review15 |
|---|---|---|---|
| PII Miss Rate ↓ | Chưa đo | Chưa đo | Chưa đo |
| Over-redaction ≤0.30 | Chưa đo | Chưa đo | Chưa đo |
| Hard slices / Recall@B | Chưa đo | Chưa đo | Chưa đo |

**Thống kê vận hành lịch sử của các run gốc** do script lấy từ run, tách khỏi điểm chất lượng. Thời gian chạy lại checkout nộp được ghi riêng trong `evidence/cascade_dev/publication_contract_checks.json`, không dùng để suy ra cải thiện thuật toán:

| Run | N ảnh | N vùng che | Tổng (s) | s/100 ảnh | ms/ảnh |
|---|---:|---:|---:|---:|---:|
| baseline | 60 | 62 | 47.9481 | 79.913 | 799.135 |
| poc | 60 | 327 | 52.6858 | 87.810 | 878.097 |
| blur_all | 60 | 60 | 12.7411 | 21.235 | 212.352 |

Bản sau review15 có **355 vùng che**, giữ nguyên **45 PNG ngoài queue**. Đây là xuất lại quyết định đã lưu, không phải inference mới; không lấy thời gian xuất lại để so detector. Blur-all là đối chứng bắt buộc và vẫn phải chấm guardrail. Bbox tăng không chứng minh recall tăng.

**Kiểm tra nội bộ trên cùng 9 ảnh dev/49 vùng người dùng đã duyệt**, không phải GT giảng viên hoặc heldout độc lập:

| Bản máy | Miss vùng / N | PII Miss Rate ↓ | Che thừa ↓ |
|---|---:|---:|---:|
| Baseline detector | 49/49 | 100,00% | 0,00% |
| PoC gốc trước review | 45/49 | 91,84% | 26,29% |
| Cascade hiện tại, nới bổ sung 0% | 2/49 | 4,08% | 28,55% |

Baseline không che trúng vùng tham chiếu trong nhóm dev này nên che thừa 0% không có nghĩa chất lượng tốt. Hai dòng đầu lấy từ [thử nghiệm cùng tham chiếu](t07_reviewed_experiment.json); dòng cascade được chạy lại từ pixel tại [metrics](evidence/cascade_dev/metrics.json). Đủ phủ 47/49 = 95,92%; chưa đạt mục tiêu bổ sung che thừa dưới 10%. Queue chọn mẫu rủi ro, nhãn có padding và đã dùng để phát triển: không suy rộng sang 27 dev chưa có nhãn, toàn eval hoặc ảnh thật.

CI95% bootstrap theo ảnh: miss 0,00–9,62%; che thừa 23,64–35,06%.

## 05 · Evaluation protocol

- **Primary / TP-FN / denominator:** PII Miss Rate = số vùng GT có hợp pixel che phủ <90% diện tích / tổng vùng GT cần che. Đơn vị là vùng; đủ coverage là hit, chưa đủ là miss; không thay bằng IoU 0.5.
- **FP / che thừa:** số pixel che ngoài hợp vùng GT / tổng pixel che, guardrail≤0.30; không đếm FP bằng IoU bbox.
- **Threshold / budget / lúc chốt:** face 0.90, plate 0.25, padding 0.03, blur sigma 12, B=15; freeze trước lần all đầu tại `2026-11-14T16:06:39.6673183+07:00`. Hash detector/model/config còn khớp; contract và pipeline hiện khác snapshot gốc do bảo trì/bổ sung. Cần freeze bản nộp hiện tại sau commit, không gán hash mới cho thời điểm cũ.
- **Heldout / CI:** tham số đầu chọn trên dev; review sau đó đã xem một số ảnh eval nên không gọi đây là test mới chưa xem. Chấm chính thức cần hidden eval mới của giảng viên. Dự kiến bootstrap theo ảnh 1.000 lần, seed 42, CI 95%; chưa tính khi thiếu nhãn độc lập.
- **Phạm vi cascade:** cổng giữ bản gốc cho 36/36 dev, chỉ 9 ảnh có nhãn được tính chất lượng. Bootstrap nội bộ theo ảnh, 1.000 lần, seed42 đã có trong `evidence/cascade_dev/metrics.json`. Trong vòng cascade không dùng val để chỉnh; lịch sử đã tiếp xúc eval trước đó vẫn được công bố. Không thay số liệu nội bộ này vào bảng metric chính thức khi chưa có overlay.
- **Lệnh evaluate:** khi có overlay GT mới chạy; hiện thiếu GT sẽ exit2 theo thiết kế.

```powershell
python -m src.evaluate --track t07 --pred reports/evidence/baseline/redactions.csv --split eval --run-name baseline --out reports/metrics_baseline.json
python -m src.evaluate --track t07 --pred reports/evidence/poc/redactions.csv --split eval --run-name poc_machine --out reports/metrics_poc_machine.json
python -m src.evaluate --track t07 --pred reports/evidence/review15/redactions.csv --split eval --run-name poc --out reports/metrics_poc.json --slices-out reports/slices_poc.csv
python -m src.evaluate --track t07 --pred reports/evidence/blur_all/redactions.csv --split eval --run-name blur_all --out reports/metrics_blur_all.json
python -m src.compare reports/metrics_baseline.json reports/metrics_poc.json --out reports/table1_baseline_vs_poc.md
```

Evaluator chuẩn có type/condition; size slices, Recall@B và CI cần script phụ khi có nhãn. Không sửa metric chuẩn. [Metric contract](metric_contract.md).

## 06 · Result slices

| Slice | N vùng GT | Baseline | PoC | Ghi chú |
|---|---|---|---|---|
| face / plate | Chưa có GT | Chưa đo | Chưa đo | Báo riêng từng loại |
| normal / small / dark / blur / occluded | Chưa có GT | Chưa đo | Chưa đo | Không suy GT condition bằng cảm giác |
| nhỏ / vừa / lớn | Chưa có GT | Chưa đo | Chưa đo | Diện tích <16², 16²–32², ≥32² px |

- **Failure 1 — pii_10 (dev):** biển số còn lộ ký tự bên phải vì bbox/padding cố định chưa phủ đủ chuỗi. Sửa bằng mở rộng vùng. [Nguồn](../demo/t07_samples/pii_10/source.jpg) · [PoC máy](../demo/t07_samples/pii_10/poc.png) · [sau review](../demo/t07_samples/pii_10/review15.png).
- **Failure 2 — pii_40 (dev):** mặt mờ bị sót, kính xe bị nhận nhầm plate. Rule mắt suy giảm khi mờ; rule hình học nhầm vùng tương phản. Sửa bằng bổ sung mặt/plate và bỏ kính. [Nguồn](../demo/t07_samples/pii_40/source.jpg) · [PoC máy](../demo/t07_samples/pii_40/poc.png) · [sau review](../demo/t07_samples/pii_40/review15.png).
- **Mặt tối — pii_08 (dev):** mặt ở phía trên trái còn thấy mắt/miệng trong ảnh máy. Giả thuyết lỗi là tương phản thấp làm rule mắt không đủ tín hiệu; bản sửa bổ sung vùng mặt. Đây là quan sát ảnh, chưa phải miss rate theo GT. [Nguồn](../demo/t07_samples/pii_08/source.jpg) · [PoC máy](../demo/t07_samples/pii_08/poc.png) · [sau review](../demo/t07_samples/pii_08/review15.png).
- **Latency / compute:** Python 3.13.17, Windows-11-10.0.22631-SP0, CPU AMD64 Family 23 Model 24 Stepping 1, AuthenticAMD. ms/ảnh ở bảng vận hành gồm xuất PNG/contact sheet, chưa đo riêng inference.

Với cascade trên tham chiếu dev: mặt 26/26 đủ phủ, biển 21/23 đủ phủ. Chưa có nhãn `condition` độc lập cho nhóm này nên không suy đoán các slice normal/dark/blur/occluded bằng tên ảnh. Size slice được tính từ diện tích bbox trong `evidence/cascade_dev/coverage_pad0.0.csv`; đây vẫn là kiểm tra dev nội bộ.

| Size slice nội bộ | N vùng | Bỏ sót | Miss rate |
|---|---:|---:|---:|
| Nhỏ, diện tích <16² px | 3 | 0 | 0,00% |
| Vừa, 16²–32² px | 22 | 1 | 4,55% |
| Lớn, ≥32² px | 24 | 1 | 4,17% |

## 07 · Human & decision

- **Queue / B / ưu tiên:** [review_queue.csv](review_queue.csv), B=15 theo risk; reviewer **Nguyễn Văn Hùng**, **14 approve + 1 edit** trên web; 9 dev/6 eval. Dùng timestamp/bbox đã lưu, không tạo lượt duyệt mới khi gom bài.
- **Override / audit:** edit thay toàn bộ danh sách bbox và xuất lại từ ảnh gốc. [Audit web](evidence/audit_web.jsonl) giữ actor, thời gian, before/after, snapshot. Bản sửa đầu có AI hỗ trợ; [journal sửa và xác nhận](evidence/audit_corrections_and_confirmation.jsonl) giữ lịch sử. [Manifest review15](evidence/review15_manifest.json) kiểm 45 ảnh ngoài queue không đổi.
- **Phạm vi nộp:** giữ queue 15 ảnh đã duyệt; 45 ảnh ngoài queue giữ kết quả PoC máy.
- **Policy / reason / owner:** **HOLD / INDEPENDENT_QA_PENDING / Nguyễn Văn Hùng**; [decision_log.csv](decision_log.csv).

## 08 · Reproduce & tests

Chạy từ repo bằng PowerShell; chuẩn bị weights trước lúc demo có mạng.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m tools.prepare_t07_models
.\.venv\Scripts\python.exe -m tools.prepare_t07_models --offline
.\.venv\Scripts\python.exe -m src.main --input fixtures/t07_redaction --config configs/lab.yaml --output out/poc --offline --seed 42
.\.venv\Scripts\python.exe -m src.evaluate --track t07 --pred reports/evidence/review15/redactions.csv --split eval --out reports/metrics_poc.json
.\.venv\Scripts\python.exe -m pytest -q
```

- **Output pytest của checkout nộp sạch:** **159 passed, 39 skipped, 0 failed**, 53,754s. [JSON](t07_current_validation.json) / [JUnit](evidence/cascade_dev/pytest_publish.xml). Bằng chứng kiểm thử cũ giữ riêng; skip cần GT/overlay giảng viên, test pass không phải điểm model.
- **Success / failure / edge:** schema/export, reviewer/snapshot/bounds sai, replay audit, pixel ngoài vùng, chặn network/GT, upload lỗi và retry khóa file Windows.
- **Máy / git sha:** máy ở mục 06; run gốc `8f1e0db`; checkout kiểm tra trước công bố bắt đầu từ `0cd85c3`. Lệnh runner trong README đã chạy lại offline, seed42, exit0 trên checkout này. Lần chạy sạch chưa có review nên giữ HOLD, không nhập giả quyết định cũ vào snapshot mới.
- **Web:** `MO_WEB_T07.cmd` để sửa ảnh; `MO_XEM_TUNG_BUOC_T07.cmd` để tải ảnh/xem 6 bước. [QA](t07_walkthrough_validation.json) đã chạy Chrome upload/model thật trên desktop và viewport 390 px.
- **Tái tạo review:** `python -m tools.prepare_t07_submission` cần run/journal local gốc; sau đó chạy `python -m tools.scope_t07_review15` để giới hạn hồ sơ theo queue15. CSV/60PNG nằm trong [ZIP](evidence/review15/results.zip).
- **Snapshot nguồn:** freeze lịch sử lưu SHA256 nguyên byte lúc đo, có file dùng CRLF trên Windows. Bản công bố Git chuẩn hóa nguồn LF; bản chụp trong `reports/evidence/` giữ nguyên byte audit/source. `reports/t07_publication.json` ghi checksum nguồn của lần công bố riêng, không sửa ngày hoặc hash của freeze cũ.

## 09 · Bảo vệ dữ liệu

- Bản nộp chỉ chứa ảnh fixture synthetic. Ảnh thật upload thử, weights và token giữ local.
- Pipeline chạy offline; web nhận ảnh qua loopback, worker dùng weights local.
- Nguồn, phiên bản, checksum và license model được ghi tại [model inventory](t07_models.json).

## 10 · Team contribution

| Học viên | Git author/email | Module / vai trò | Commit chứng minh | Trình bày? |
|---|---|---|---|---|
| Nguyễn Văn Hùng — 2A202603097 | hunghocweb / hunghocweb2709@gmail.com | src/t07_*, config, tests, tools, review, report | `8f1e0db` code; mã web và hồ sơ tại commit cuối (`git log -1`) | Chưa trình bày demo |

Đối chiếu `git shortlog -sne --all` với [contributions](team_contributions.csv). Code/đề xuất sửa có Codex hỗ trợ; quyết định web mục 07 do Nguyễn Văn Hùng đã lưu.

## 11 · References

| URL / tài liệu | Ngày xem | Ý nghĩa |
|---|---|---|
| [README và template nhánh main](https://github.com/hunghocweb/K5-Day6-L2-NguyenVanHung-2A202603097-DataGovernanceAndPrivacy/tree/main/reports) | 15/11/2026 | Đối chiếu mục00–11 và checklist; main được đọc tại `b0ce8bf`. |
| [Đề T07](../docs/tracks/T07.md) | 15/11/2026 | Output, coverage 90%, che thừa ≤30%, budget 15, hard negative và failure modes. |
| [README gốc VinUni](https://github.com/VinUni-AI20k/K5-Day6-L2-DataGovernanceAndPrivacy/blob/main/README.md) | 16/11/2026 | Mục 2–3 về lệnh chấm/cấu trúc, mục 4–7 về báo cáo, human loop và rubric. |
| [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | 14/11/2026, theo inventory | Detector mặt CPU, nguồn và license. |
| [open-image-models](https://github.com/ankandrew/open-image-models) / [issue license](https://github.com/ankandrew/open-image-models/issues/14) | 14/11/2026, theo inventory | Detector plate ONNX và giới hạn xác nhận license weights. |

**Bảng 3 — Checklist chốt khi nộp**

- [x] Config/provenance/seed/model inventory; baseline/PoC/blur-all; hai failure cases có ảnh.
- [x] Queue 15 có reviewer/override/audit/decision; ZIP 60 ảnh, giữ 45 ảnh ngoài queue.
- [x] Offline/tests đạt; README, metric contract và contributions đúng người/commit đã có.
- [ ] Chấm baseline/PoC/blur-all, hard slices và bảng so sánh bằng script khi có nhãn hợp lệ.
- [x] Có bộ 5 slide [HTML](../demo/T07_G07_slides.html) / [PDF](../demo/T07_G07_slides.pdf) và [kịch bản](../demo/showcase.md).
- [ ] Trình bày demo.
- [x] Mã nguồn và hồ sơ T07 đã push; có tag `demo-180m` và `final-24h`.
