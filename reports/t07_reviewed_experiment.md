# Thử nghiệm T07 với bbox đã được người dùng duyệt

## 1. Nhãn tham chiếu lấy từ đâu?

Dùng đúng bbox `after` của quyết định human mới nhất trong queue 15 ảnh: 14 approve và 1 edit. Giữ nguyên split gốc: 9 dev và 6 eval (gọi là val trong yêu cầu). Không dùng 45 ảnh ngoài queue làm ảnh âm hoặc tự sinh nhãn còn thiếu.

Loại pii_00 vì edit có ghi chú chưa xử lý hết; pii_02 và pii_03 đang reject. Không ghi đè audit/review hoặc GT giảng viên.

Đây là **GT nội bộ từ vùng che đã duyệt**, có thể chứa padding và sai sót review. Nhãn bắt đầu từ dự đoán máy rồi được sửa/duyệt, nên có nguy cơ thiên lệch về hệ thống gốc. Không xem các số này là điểm chính thức hoặc bằng chứng tổng quát hóa ảnh thật.

## 2. Cách đo và khóa cấu hình

Seed 42. Miss khi tổng mask che phủ <90% một vùng tham chiếu; success = 1 − miss. Che thừa = pixel che ngoài hợp vùng tham chiếu / tổng pixel che, guardrail ≤30%. Dùng hàm metric của repo; không thay bằng IoU hoặc confidence. CI bootstrap theo ảnh, 1000 lần, seed 42. Không có nhãn condition nên không suy đoán slice condition.

Nguồn v3/router và tham số được hash trước khi đo dev; chạy dev trước, sau đó eval với cùng cấu hình, không tune theo kết quả. Router vẫn dùng ngưỡng thử 0,60 ±0,08; chưa hiệu chỉnh thật/2D vì không có ảnh thật. v3 padding 1% ở cả hai nhánh so sánh router.

Baseline và PoC v1 là dự đoán máy đã lưu, không áp sửa human. Hai cấu hình v3 được chạy mới từ ảnh nguồn. Không chấm bản review15 với chính nó rồi gọi là chất lượng máy.

## 3. Kết quả

| Tập | Cấu hình | Ảnh / vùng nhãn | Vùng phủ ≥90% | Bỏ sót | Che thừa | Guardrail |
|---|---|---:|---:|---:|---:|---|
| dev | baseline | 9 / 49 | 0.00% | 49/49 (100.00%) | 0.00% | PASS |
| dev | poc_v1 | 9 / 49 | 8.16% | 45/49 (91.84%) | 26.29% | PASS |
| dev | poc_v3 | 9 / 49 | 48.98% | 25/49 (51.02%) | 35.22% | FAIL |
| dev | poc_v3_router | 9 / 49 | 48.98% | 25/49 (51.02%) | 35.22% | FAIL |
| eval | baseline | 6 / 31 | 0.00% | 31/31 (100.00%) | 0.00% | PASS |
| eval | poc_v1 | 6 / 31 | 51.61% | 15/31 (48.39%) | 0.00% | PASS |
| eval | poc_v3 | 6 / 31 | 90.32% | 3/31 (9.68%) | 47.08% | FAIL |
| eval | poc_v3_router | 6 / 31 | 90.32% | 3/31 (9.68%) | 47.08% | FAIL |

V3 tăng tỷ lệ vùng phủ đủ so với PoC v1 thêm 40,82 điểm phần trăm trên dev và
38,71 điểm phần trăm trên eval. Tuy vậy **chưa đạt yêu cầu**, vì cả hai tập đều
FAIL guardrail che thừa. Không chọn cấu hình thắng chỉ từ miss rate thấp hơn.

Với v3, ảnh dev cần xem lại nhiều nhất là pii_24 (5/7 vùng còn thiếu), pii_15
(4/5), pii_12 (4/5), pii_19 (4/5). Trên eval, mỗi ảnh pii_52, pii_33, pii_56
còn một vùng chưa đạt coverage 90%. Đây là chẩn đoán sau đo, chưa dùng để sửa
rule hoặc chạy lại kết quả eval trong thử nghiệm này.

### Theo mặt / biển số

| Tập | Cấu hình | Loại | Vùng bỏ sót / nhãn | Miss rate |
|---|---|---|---:|---:|
| dev | baseline | face | 26/26 | 100.00% |
| dev | baseline | plate | 23/23 | 100.00% |
| dev | poc_v1 | face | 22/26 | 84.62% |
| dev | poc_v1 | plate | 23/23 | 100.00% |
| dev | poc_v3 | face | 10/26 | 38.46% |
| dev | poc_v3 | plate | 15/23 | 65.22% |
| dev | poc_v3_router | face | 10/26 | 38.46% |
| dev | poc_v3_router | plate | 15/23 | 65.22% |
| eval | baseline | face | 16/16 | 100.00% |
| eval | baseline | plate | 15/15 | 100.00% |
| eval | poc_v1 | face | 0/16 | 0.00% |
| eval | poc_v1 | plate | 15/15 | 100.00% |
| eval | poc_v3 | face | 1/16 | 6.25% |
| eval | poc_v3 | plate | 2/15 | 13.33% |
| eval | poc_v3_router | face | 1/16 | 6.25% |
| eval | poc_v3_router | plate | 2/15 | 13.33% |

### Router

Route trên toàn bộ 60 ảnh: `{"2d": 60}`.

Route là dự đoán từ pixel. Tất cả fixture được mô tả là synthetic trong tài liệu, nhưng không sử dụng trường này để định tuyến. Nếu router và v3 cho cùng kết quả trên tập này, chưa có bằng chứng router tăng chất lượng; vẫn cần ảnh thật/ảnh lai để kiểm tra.

## 4. Giới hạn xác thực

Chỉ 9/36 ảnh dev và 6/24 ảnh eval có nhãn được duyệt trong thử nghiệm. Queue chọn ảnh rủi ro cao, không phải mẫu ngẫu nhiên đại diện. Không báo số này như kết quả toàn tập.

Ảnh trong cả hai nhóm đã được xem/review trước khi xây v3; đặc biệt pii_03 eval từng được dùng để phân tích lỗi, và các bản sửa/ảnh review khác cũng có thể đã ảnh hưởng thiết kế rule. Không chứng nhận 6 ảnh eval là holdout độc lập. Lần chạy này chỉ giữ nguyên tham số giữa dev và eval; không xóa việc đã tiếp xúc dữ liệu trước đó.

Để xác thực độc lập cần nhãn giảng viên hoặc tập ảnh mới chưa dùng chỉnh thuật toán. Không nhận bonus ablation hoặc kết luận hệ thống an toàn chỉ từ thử nghiệm này.

## 5. File bằng chứng và tái lập

- `reports/t07_reviewed_experiment.json`: số đếm, CI, per-image, timing và provenance.
- `reports/evidence/reviewed_reference/regions.csv`: bbox tham chiếu nội bộ.
- `reports/evidence/reviewed_reference/provenance.json`: quyết định gốc và hash audit.
- `reports/evidence/reviewed_reference/experiment_freeze.json`: hash nguồn/tham số trước dev.
- Cùng thư mục bằng chứng có bản sao dự đoán của bốn cấu hình, coverage từng
  vùng, `routes.json`, audit định tuyến riêng và JUnit 17 test đã chạy đạt.
- `out/t07_reviewed_experiment/`: dự đoán, coverage mỗi vùng, route/audit của lần chạy.

```powershell
.\.venv\Scripts\python.exe -m tools.evaluate_t07_reviewed
```
