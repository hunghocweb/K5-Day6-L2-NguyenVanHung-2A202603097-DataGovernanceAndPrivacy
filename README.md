# Day 06 · Data Lab · Field-to-Release PoC (AI20K · Data Track)

Starter repo cho bài lab **Data Pipeline, Privacy & Production PoC**: 14 challenge (T01–T14), một contract chung.
Repo cung cấp **dữ liệu offline + hạ tầng dùng chung + metric chuẩn + template báo cáo** để nhóm tập trung vào phần quan trọng:
vấn đề, baseline, thuật toán, đánh giá trung thực, human-in-the-loop và quyết định GO/HOLD/REWORK.

> Repo **không** chứa lời giải. `src/pipeline.py` là file nhóm sở hữu; mọi thuật toán (kể cả baseline) do nhóm tự cài.
> Tài liệu học tập, không thay tư vấn pháp lý.

## 1. Quickstart (≤ 5 phút, chạy offline)

```bash
# Python 3.11–3.13 (đã test trên 3.13)
make setup                 # = python3 -m venv .venv && pip install -r requirements.txt
make preflight             # kiểm tra môi trường, fixtures, offline guard, không lộ token
make test                  # pytest -q  (phải 0 failed)

# Ví dụ wiring E2E (random baseline, có cho T01/T02/T11/T13):
make baseline TRACK=t11      # ghi out/baseline/; repo học viên không có gt/ nên bước chấm
                             # (make eval-baseline) chỉ chạy được khi đã cài overlay giảng viên, nếu không exit 2
```

Chọn challenge: sửa `configs/lab.yaml` (`track`, `track_config`, `owner`, `group_id`), đọc `docs/tracks/TXX.md`.

### Bản nộp G07 chọn T07

Owner: Nguyễn Văn Hùng, MSSV 202603097. Windows PowerShell dùng Python trực tiếp:

Hồ sơ hiện tại: [báo cáo theo template Day06](reports/group_report.md), [checklist và danh sách file cần nộp](reports/submission_checklist.md), [gói kết quả review15](submissions/T07_G07_review15_candidate.zip). Trạng thái **REWORK**: còn hai reject và chưa có metric GT độc lập; tag demo/final chưa tạo. Bản GitHub `main` tại lần đối chiếu 15/11/2026 vẫn là starter; các thay đổi đang được hoàn thiện local.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Chuẩn bị có mạng trước demo; model tải về phải khớp inventory SHA256:
.\.venv\Scripts\python.exe -m tools.prepare_t07_models
.\.venv\Scripts\python.exe -m tools.preflight
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.main --config configs/lab.yaml --track t07 --output out/poc --run-name poc --offline --seed 42
```

Trên macOS/Linux: `make setup`, `.venv/bin/python -m tools.prepare_t07_models`, rồi `make preflight test run`.
`make baseline TRACK=t07` dùng `src.t07_baseline`; đối chứng: `python -m src.main --track t07 --pipeline src.t07_blur_all --output out/blur_all --offline --seed 42`.
Khi demo offline, `python -m tools.prepare_t07_models --offline` chỉ kiểm weights có sẵn; pipeline không tải model.

Hướng dẫn review và replay audit: [T07 runbook](docs/T07_RUNBOOK.md).

### Web review T07 tại máy

**Bấm đúp `MO_WEB_T07.cmd` ở thư mục gốc repo** để tự khởi động server và mở trình duyệt.

Sau khi đã chạy PoC, mở giao diện thao tác trực tiếp:

```powershell
.\.venv\Scripts\python.exe -m tools.t07_review_web
```

Truy cập **http://127.0.0.1:8765**. Chọn ảnh → so sánh baseline/PoC/bản sửa → vẽ, kéo, đổi kích thước hoặc xóa vùng che → nhập reviewer và quyết định → **Lưu**.
Nút **Xuất ảnh & nhật ký** tải ZIP gồm 60 ảnh PNG, `redactions.csv`, review CSV, audit và quyết định.
Phiên web lưu riêng tại `out/web_review`, bắt đầu từ `out/poc_ai_corrected` nếu có, nếu không dùng `out/poc`.
Lượt duyệt cũ không tự chuyển thành quyết định mới trên web. Chi tiết: [hướng dẫn web](docs/T07_WEB_REVIEW.md).

**Tải ảnh khác và xem thuật toán chạy từng bước:** bấm đúp `MO_XEM_TUNG_BUOC_T07.cmd`, mở [trang xem từng bước](http://127.0.0.1:8765/walkthrough). Chọn ảnh → tiền xử lý → phát hiện → gộp trùng → nới vùng → blur; có ảnh trung gian và tải kết quả PNG/CSV/ZIP. Xử lý local bằng model có sẵn. Chi tiết và giới hạn: [hướng dẫn walkthrough](docs/T07_WALKTHROUGH.md).

Kết quả máy giữ tại `out/poc`; đề xuất sửa ban đầu giữ tại `out/ai_review` và `out/poc_ai_corrected`. Nguyễn Văn Hùng đã lưu 14 approve và 1 edit cho queue15 trên web; [review_queue.csv](reports/review_queue.csv) và các journal giữ riêng nguồn sửa/duyệt. Ba lượt web ngoài queue được ghi riêng, không áp vào gói review15. Các vùng này không được coi là GT độc lập.
Nguồn/checksum/license: `reports/t07_models.json`; giới hạn license weights YOLO được ghi tại đó.

```powershell
# Chỉ áp các đề xuất bbox AI đã ghi vào đúng pool/snapshot đã review:
.\.venv\Scripts\python.exe -m tools.apply_t07_ai_corrections
```

Các số đo chất lượng chính thức vẫn cần GT của giảng viên. Báo cáo kết quả và giới hạn: `reports/group_report.md`.

## 2. Contract lệnh (giảng viên sẽ chạy đúng các lệnh này)

| Lệnh | Điều kiện PASS |
|---|---|
| `python -m src.main --input fixtures/<track> --config configs/lab.yaml --output out/` | exit 0, sinh output + review log + decision |
| `python -m src.evaluate --gt fixtures/<track>/gt/<file> --pred out/<pred_file>` | giảng viên chạy (repo học viên không có `gt/`); in baseline/PoC và định nghĩa số mẫu, tính bằng script |
| `pytest -q` | ≥2 test của nhóm (success + failure/edge), 0 failed |
| `python -m src.main --offline --seed 42` | không network; thiếu model → fail rõ ràng, không tự download |
| `git shortlog -sne --all` | mỗi người có tên/email, khớp `reports/team_contributions.csv` |

Exit code của `src.main`: `0` OK · `2` input/output sai · `3` chưa implement · `4` gọi network khi `--offline` · `5` pipeline đọc ground truth.

`src.main` tự bật hai guard: **offline guard** (chặn socket ra ngoài) và **GT guard** (chặn đọc `fixtures/*/gt/`).
Ground truth **không có trong repo học viên**: `fixtures/*/gt/`, generator và hidden eval nằm ở repo private của giảng viên.
Giảng viên chép chúng vào khi chấm (`src.evaluate`; với T12, nhãn ẩn đọc qua `AuditOracle` có budget). Học viên tự thiết kế
kiểm tra nội bộ (ví dụ tự gán nhãn vài mẫu) và ghi rõ trong report.

## 3. Cấu trúc repo

```
README.md · requirements.txt · pyproject.toml · Makefile · run.sh
configs/lab.yaml               # config của nhóm (chọn track, owner, seed)
configs/tracks/tXX.yaml        # default params + eval params cho từng track
src/main.py                    # one-command runner (contract)  — KHÔNG cần sửa
src/pipeline.py                # ★ PIPELINE CỦA NHÓM: load → process → rank → review → decide
src/evaluate.py · src/compare.py
src/examples/random_baseline.py  # ví dụ nối dây E2E (random = baseline của T01/T02/T11/T13)
src/common/                    # hạ tầng dùng chung (xem bảng dưới)
src/metrics/                   # thư viện metric + evaluator chuẩn cho 14 track (tracks.py)
fixtures/<track>/              # dữ liệu synthetic offline, KHÔNG có PII thật; gt/ chỉ có ở repo giảng viên
data/real/                     # (gitignored) mẫu dữ liệu thật do giảng viên tải trước
docs/tracks/T01..T14.md        # đề bài chi tiết + data schema + metric + output contract
docs/INSTRUCTOR_SETUP.md       # chuẩn bị gói offline P0–P6, dữ liệu thật từ Hugging Face
reports/                       # group_report.md, metric_contract.md, team_contributions.csv, metrics_*.json
tests/                         # test hạ tầng + metric; nhóm thêm test của mình ở đây
tools/                         # download_real_samples, verify_manifests, preflight (generator ở repo giảng viên)
demo/                          # slides / ảnh chụp demo
```

### Hạ tầng dùng chung (`src/common/`)

| Module | Dùng để |
|---|---|
| `io.py` | config YAML, `set_seed`, sha256, JSON, git sha |
| `manifest.py` | `manifest.csv` (file_id, path, source, license, split, group_id, sha256): build + verify |
| `offline.py` / `gt_guard.py` | chặn network / chặn đọc ground truth |
| `review.py` | review queue top-B, đọc quyết định reviewer, `apply_overrides`, `AuditLog` (JSONL append-only) |
| `contact_sheet.py` | contact sheet HTML offline (ảnh trước/sau, bbox, dropdown quyết định, nút **Export CSV**) |
| `decision.py` | `Decision` GO/HOLD/REWORK (+ alias PASS/ACCEPT/KEEP/RECAPTURE…) bắt buộc `reason_code` + `owner` |
| `report.py` | ghi `metrics_*.json` (seed, git sha, runtime/100 items, máy) + bảng Baseline vs PoC |
| `corruptions.py` | blur / dark / glare / jpeg / fog / rain / night / tint với severity 0–2 (deterministic) |
| `image_ops.py` | Laplacian variance, brightness stats, histogram, aHash/dHash/pHash + Hamming |
| `audit_oracle.py` | "người kiểm" mô phỏng: chỉ lộ nhãn ẩn cho mẫu được chọn, dừng khi hết budget |
| `timer.py` | đo thời gian, ms/item, thông tin máy |

### Metric chuẩn (`src/metrics/`)

Mỗi track có evaluator riêng trong `src/metrics/tracks.py`, cài đúng primary metric trong handout:

| Track | Primary metric | Output nhóm phải ghi |
|---|---|---|
| T01 Coverage Planner | Weighted Coverage@K (K=30) | `pick.csv` |
| T02 Raw QC Triage | Defect Recall@B + Precision@B (B=20) | `ranked_review.csv` |
| T03 Synthetic-vs-Real | Transform QC F1 | `qc.csv` (+ `transform_manifest.csv`) |
| T04 Sensor Sync Guard | Pair Accuracy@50ms | `pairs.csv` |
| T05 Leakage & Near-Dup | Leakage Detection F1 (pair) | `suspect_pairs.csv` (+ `split_manifest.csv`) |
| T06 Integrity & Provenance Gate | Silent-defect Detection F1 (release) | `release_decisions.csv` |
| T07 Face/Plate Redaction | PII Miss Rate (coverage ≥ 0.9) + guardrail over-redaction ≤ 0.3 | `redactions.csv` |
| T08 Delete Request Simulator | Deletion Completeness (+ false-delete = 0) | `actions.csv` (+ `audit_log.jsonl`) |
| T09 Privacy Decision Gate | Critical-miss Rate (split eval) + guardrail false HOLD ≤ 0.5 | `decisions.csv` |
| T10 Domain Shift Radar | Spearman ρ(ShiftScore, PerfDrop) + bootstrap CI | `shift_scores.csv` |
| T11 Label Error Prioritizer | Error Recall@B, B=max(10, 10%N) | `ranking.csv` |
| T12 Vendor Batch Acceptance | MAE + Bad-batch Recall (báo riêng) + guardrail HOLD rate ≤ 0.4 | `batch_decisions.csv` |
| T13 Rare Case Miner | Rare-case Recall@K (K=20) | `selected.csv` |
| T14 Auto-Annotation + QC | Detection F1@IoU0.5 | `predictions.csv` |

Thư viện chung (`classification`, `ranking`, `detection`, `pairs`, `timing`, `estimation`, `correlation`) có docstring ghi rõ numerator/denominator.
**Không sửa định nghĩa metric để đẹp số.** Nếu thấy metric chưa hợp lý, hãy lập luận trong report và thêm metric phụ.

## 4. Quy trình làm bài (240 phút)

| Phút | Việc | Checkpoint |
|---|---|---|
| 00–10 | Chọn challenge, lập repo/nhóm, chia module | mỗi người 1 commit đầu |
| 10–30 | Đọc `docs/tracks/TXX.md`, chốt GT + metric | điền `reports/metric_contract.md` |
| 30–50 | Baseline + 2–3 fixture lỗi đầu tiên | `reports/metrics_baseline.json` |
| 50–145 | Pipeline xử lý → review → report | one-command PoC + `review_queue.csv` / contact sheet |
| 145–170 | Heldout, ablation nhanh, sửa bug | `reports/metrics_poc.json` + 2 failure cases |
| 170–180 | Freeze demo | `make tag-demo` (tag `demo-180m`), 3–5 slides |
| 180–240 | Showcase + adversarial debate | |
| +24h | Repo tái lập, report, contributions | `make tag-final` (tag `final-24h`) |

Lệnh nộp số liệu (không gõ tay):

```bash
python -m src.main --track tXX --output out/baseline --run-name baseline --pipeline <module baseline của nhóm> --offline
python -m src.main --track tXX --output out/poc --run-name poc --offline
make eval-baseline eval-poc compare TRACK=tXX      # cần overlay giảng viên (gt/); không có thì exit 2
                                                   # -> reports/metrics_*.json + reports/table1_baseline_vs_poc.md
```

### Bắt đầu viết pipeline (15 phút đầu)

1. Sửa `configs/lab.yaml`: `group_id`, `owner`, `track: tXX`, `track_config: tracks/tXX.yaml`.
2. Đọc `docs/tracks/TXX.md`, mở `fixtures/<track>/` xem input; xem `src/examples/random_baseline.py` để biết cách nối dây.
3. Trong `src/pipeline.py`, viết `load()` + `process()`; `process()` trả DataFrame có **các cột bắt buộc** của track
   (bảng trên) **và** cột `score` (số, cao = cần review trước) để `rank()` lập queue top-B.
4. `make run TRACK=tXX` (exit 0). Exit 3 nghĩa là `process()` chưa được cài. `make eval-poc` cần `gt/` của giảng viên
   (repo học viên trả exit 2); tự kiểm bằng metric không cần nhãn và review thủ công.

### Vòng human-in-the-loop (review → override → audit)

1. `make run` sinh `out/poc/review_queue.csv` (top-B). Muốn có giao diện: gọi
   `render_contact_sheet(queue, ctx.out("contact_sheet.html"), image_root=ctx.input_dir)` (xem `random_baseline.py`), mở file HTML bằng trình duyệt (offline).
2. Reviewer nhập tên, chọn quyết định (`approve/reject/edit/escalate/skip`), bấm **Export CSV**
   → trình duyệt tải về `review_queue_reviewed.csv`. Hoặc điền thẳng các cột `reviewer`, `reviewer_decision`, `override_value`, `review_note` trong CSV.
3. Chép file vào **`out/poc/review_queue_reviewed.csv`** rồi chạy lại `make run`: bước `review()` đọc file
   (dòng có quyết định mà thiếu `reviewer` → fail rõ), nhóm tự áp override (`common.review.apply_overrides`) và ghi `out/poc/audit_log.jsonl`.

## 5. Dữ liệu

- `fixtures/`: **synthetic**, CC0, không có người/biển số thật; dữ liệu input đã commit sẵn.
  Mỗi track có `manifest.csv` (input) kèm sha256. Kiểm tra: `make verify`.
  Ground truth (`gt/`) và generator không phát cho học viên; `pytest` tự skip các test cần GT.
- `data/real/` (gitignored, tuỳ chọn): mẫu nhỏ dữ liệu thật từ Hugging Face (BDD100K val, COCO val2017, license plates).
  Giảng viên chuẩn bị trước lớp bằng `make real-data` — xem `docs/INSTRUCTOR_SETUP.md`.
  **Có người và biển số thật → không commit, không đưa lên cloud/public repo.**
- Dữ liệu thật **không có GT ẩn** do giảng viên giữ: nếu dùng, nhóm tự khai báo protocol GT trong report.
  Điểm chính vẫn tính trên fixtures (có hidden eval tương đương).

## 6. Quy tắc

1. Không đọc `gt/` trong pipeline; không tune trên split eval/heldout; chốt threshold/trọng số trước.
2. Mọi run dùng `--seed 42`, ghi N và tốc độ (đã có trong `run_info.json` / `metrics_*.json`).
3. Không gọi API ngoài khi demo; model pretrained phải tải trước, ghi tên/checksum/version/license.
4. Không đưa ảnh người thật/biển số thật chưa có quyền lên repo công khai hoặc cloud.
5. Không tuyên bố "đạt GDPR / luật Việt Nam" từ checklist tự động; nêu giả định cần DPO/legal xác nhận.
6. Mỗi thành viên ≥1 commit thực chất (khuyến nghị 2); commit chỉ đổi khoảng trắng không tính.
7. Không đổi testset hoặc chèn kết quả không tái lập sau khi chốt protocol.

## 7. Rubric (100 điểm + bonus ≤ 15)

| Tiêu chí | Điểm |
|---|---|
| End-to-end & đúng yêu cầu (1 lệnh, 5 bước, demo thật) | 22 |
| Gold set + metric rõ | 18 |
| Baseline + kết quả so sánh (cùng budget/heldout/seed, trung thực khi 0/âm) | 18 |
| Human-in-loop & risk gate (queue, override, audit log, reason/owner) | 12 |
| Reliability & reproducibility (offline, clean start, ≥2 tests, 1 failure mode) | 12 |
| Report + team Git evidence | 10 |
| Showcase & debate | 8 |

Bonus: +5 verified E2E trước phút 180 (tag) · +5 ablation/hidden test thật · +5 debate tìm bug thật bằng test.
