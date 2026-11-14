# Kiểm tra hồ sơ nộp Day06 — G07 / T07

Hồ sơ gồm mã nguồn và bằng chứng T07 theo cấu trúc mục3 của README VinUni.

## 1. Cấu trúc file nộp

| Thư mục | Nội dung |
|---|---|
| configs/ | lab.yaml, trackT07 và profile dev mở rộng |
| src/ | pipelineT07, baseline/blur-all, detector/ba nhánh và web |
| tests/ | testT07 về success, failure/edge, audit, export và fallback |
| tools/ | chuẩn bị weights, chạy/kiểm tra/demo và xuất báo cáo |
| docs/ | hướng dẫnT07; tài liệu đề gốc giữ nguyên |
| reports/ | group_report mục00–11, metric_contract, contributions, queue, decision và bằng chứng |
| demo/ | 5slide HTML/PDF, kịch bản và ảnh failure case synthetic |

[Danh sách từng đường dẫn](submission_files.txt) là phạm vi được chọn; không dùng `git add .`. Gói60PNG cuối và audit tại [evidence/review15/results.zip](evidence/review15/results.zip). README/Makefile/run.sh/entrypoint và hạ tầng có sẵn vẫn giữ từ commit cha; không cập nhật các file ngoài phạm vi.

## 2. Đã có bằng chứng

- G07, Nguyễn Văn Hùng, MSSV202603097; configT07, seed42, split36dev/24eval, budget review15.
- Pipeline đúng5bước: detect → che → queue → human sửa → export/audit/decision; có baseline và đối chứng blur-all.
- Queue15:14approve+1edit đã lưu; 45ảnh ngoài queue giữ kết quả PoC máy. Quyết định HOLD/INDEPENDENT_QA_PENDING.
- Cascade mở rộng giữ2D elip/mờ gốc, dùng2D chi tiết dự phòng và model cho ảnh thật. Nhánh2D không gọi model ảnh thật.
- Kiểm tra nội bộ cascade:9dev/49vùng có nhãn,47/49đủ phủ, che thừa28,55%.
- Checkout nộp sạch:159passed,39skipped,0failed; [JUnit](evidence/cascade_dev/pytest_publish.xml), [JSON](t07_current_validation.json).39skip do thiếu overlayGT. Lệnh runner README đã chạy offline/seed42, exit0, lần chạy mới giữHOLD vì chưa có human review.
- Bộ5slide [HTML](../demo/T07_G07_slides.html)/[PDF](../demo/T07_G07_slides.pdf); chưa trình bày demo.
- Có tag `demo-180m` và `final-24h` cho bản nộp.

## 3. Chưa đủ để chốt toàn bộ lab

1. Chưa có GT/hidden overlay chính thức, nên chưa có `metrics_baseline.json`, `metrics_poc.json`, `metrics_blur_all.json` và `table1_baseline_vs_poc.md` do evaluator chuẩn sinh. Không điền số giả hoặc lấy metricdev thay cho eval chính thức.
2. Trình bày demo.

## 4. Phần giữ local

Không đưa `.venv/`, `out/`, cache, `.serena/`, `.env`/token, weights, `data/real/`, `gt/` hoặc ảnh thật upload thử lên GitHub. Nguồn/phiên bản/checksum/license của model nằm trong `t07_models.json`. Byte audit và snapshot lịch sử trong `evidence/` giữ nguyên; manifest công bố ghi riêng checksum Git để không sửa freeze cũ khi chuẩn hóa LF.

README gốc, fixtures và hạ tầng chấm giữ nguyên.
