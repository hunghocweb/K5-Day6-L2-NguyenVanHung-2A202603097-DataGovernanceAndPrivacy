# Metric contract (khai báo bắt buộc — chốt ở phút 30, trước khi chạy eval)

> Các metric chuẩn đã được cài trong `src/metrics/tracks.py`. File này là nơi nhóm **cam kết** cách dùng chúng.
> Không đổi testset, threshold hay trọng số sau khi đã thấy kết quả eval.

| Mục | Khai báo của nhóm |
|---|---|
| Track | T07 Face/Plate Redaction |
| Primary metric (tên đúng như evaluator in ra) | PII Miss Rate; kèm Over-redaction rate và guardrail ≤ 0.30 |
| Tăng tốt hay giảm tốt? | lower cho miss và over-redaction |
| Đơn vị đánh giá (image / region / pair / release / batch / case …) | Vùng PII cho miss; pixel hợp nhất cho over-redaction |
| Công thức + numerator / denominator | Miss = vùng must_redact có coverage < 0.90 / tổng vùng must_redact. Over-redaction = pixel che ngoài hợp vùng GT / tổng pixel che. |
| Cách ghép prediction ↔ GT (IoU? id? tolerance?) | Ghép image_id, coverage bằng hợp mask pixel; không dùng IoU 0.5 để thay metric. |
| Candidate pool cố định | 60 ảnh fixture 640×360, 36 dev và 24 eval; không sửa split/input. Báo N vùng từ script khi có nhãn. |
| Review budget B / K | 15 ảnh ưu tiên theo risk, thứ tự ổn định; báo riêng kết quả máy và sau human correction. |
| Seed | 42 |
| Threshold policy (chọn trên dev như thế nào) | Freeze 2026-11-14 16:05 +07:00, trước lần chạy all đầu tiên: face confidence 0.90, plate 0.25, padding PoC 0.03, baseline 0, blur sigma 12, B=15. Rule hình học và risk cố định trong src/t07_detectors.py và src/t07_pipeline.py. Chỉ xem ảnh dev và probe dev trước freeze; không dùng nhãn eval. |
| Eval split (không dùng để tune) | eval: 24 ảnh. GT giảng viên chưa có; không tự sinh hoặc suy đoán hidden GT. |
| Chỉ số lỗi xấu (miss / false alarm / false reject) | Bỏ sót PII, che thừa, đặc biệt đầu quay lưng bị che nhầm; không dùng confidence thay nhãn thật. |
| Slices sẽ báo cáo (≥2) | face/plate; normal/small/dark/blur/occluded; kích thước vùng (nhỏ <16² px, vừa 16²–32² px, lớn ≥32² px, bổ sung ngoài evaluator chuẩn). |
| Cách báo độ bất định khi N nhỏ (Wilson CI / bootstrap) | Bootstrap theo image_id, 1000 lần, seed 42; báo 95% percentile CI và số ảnh/vùng. CI nhỏ mẫu chỉ dùng mô tả, không suy rộng sang ảnh thật. |
| Ngày giờ chốt + người chốt | 2026-11-14 16:05 +07:00, Codex ghi nhận protocol kỹ thuật theo kế hoạch được giao; G07, owner Nguyễn Văn Hùng, MSSV 202603097. Đây không phải phê duyệt release hay review ảnh của owner. Snapshot hash: reports/t07_protocol_freeze.json. |

Nếu baseline = 0 thì ghi rõ "baseline = 0" thay vì phần trăm cải thiện.

## Kiểm tra nội bộ và human-in-the-loop

- Tại lần khóa protocol gốc chưa có GT nội bộ. Hiện có tham chiếu từ bbox người dùng đã duyệt, được khai báo riêng dưới đây; không coi là GT giảng viên hoặc heldout độc lập.
- Nếu xây tập kiểm nội bộ: chọn mẫu dev trước, người thật gán bbox face/plate và phân biệt đầu quay lưng; lưu tên, ngày, hướng dẫn và phạm vi. Không đưa ảnh eval vào quá trình tune.
- Mọi review cần reviewer thật, quyết định và ghi chú; edit thay toàn bộ danh sách vùng che của ảnh để hỗ trợ thêm/sửa/xóa. Chạy lại từ ảnh gốc và dự đoán máy, giữ before/after cùng audit.
- Không tự đánh dấu approve. Skip/escalate/chưa có reviewer giữ risk unresolved; thiếu owner/group_id giữ HOLD. Hoàn tất queue không chứng minh toàn bộ batch không bỏ sót.
- Chính sách đã cài: reject có ghi chú nhưng chưa có bbox sửa → REWORK/HUMAN_CORRECTIONS_REQUIRED và tạo human_change_requests.csv; còn item chưa duyệt → HOLD/HUMAN_REVIEW_PENDING; queue xong nhưng chưa có chứng cứ QA độc lập → HOLD/INDEPENDENT_QA_PENDING; dữ liệu review sai → fail rõ. Không tự GO hoặc tuyên bố an toàn từ việc duyệt top-15. Không dùng hidden GT trong pipeline để quyết định.

## Bản mở rộng dev — công bố 16/11/2026

- Giữ nguyên protocol/PoC gốc ở trên để tái lập review15. `configs/lab_auto_dev.yaml` là profile riêng của `auto-cascade-1`: chỉ dev, seed42, B15, nới bổ sung0%; nhánh `math2d-dev-1` giữ margin gốc.
- Bộ tham chiếu nội bộ có 9 ảnh dev/49 vùng từ queue đã duyệt, lưu tại `reports/evidence/reviewed_reference/regions_dev.csv`. Nhãn có thể chứa padding và ảnh hưởng từ dự đoán máy ban đầu. Nhóm đã phát triển trên các mẫu này; không gọi kết quả là kiểm tra độc lập hoặc điểm của toàn36dev.
- Cổng chọn nhánh từ pixel: bảng màu tập trung và mật độ cạnh; không dùng tên ảnh/ID/split/manifest trong detector. Đầu ra rỗng hoặc ảnh nhiều chi tiết chuyển sang bộ2D chi tiết. Điểm cổng không phải accuracy90% của ảnh mới.
- Số liệu, numerator/denominator, type/size slice và CI bootstrap theo ảnh được sinh bằng `python -m tools.verify_t07_cascade_dev --reviewed-only`, lưu tại `reports/evidence/cascade_dev/`. Không dùng val để chỉnh trong vòng cascade.
- Không đổi các mốc freeze lịch sử và không nhận là đã có GT chính thức, blind holdout, hoặc đã đạt che thừa<10%. Bảnreview15 và output mới giữ tách biệt; lần chạy mới chưa có người duyệt giữ HOLD.
