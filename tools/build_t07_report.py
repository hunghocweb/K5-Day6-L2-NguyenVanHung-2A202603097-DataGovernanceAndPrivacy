"""Render Day06 sections 00-11 from recorded submission evidence."""
import json
from tools.prepare_t07_submission import git
from tools.t07_review_web import ROOT, csv_bytes


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def main():
    a=read('reports/t07_submission_audit.json'); r=a['review15']; t=a['tests']
    runs=read('reports/t07_run_summary.json')['runs']; info=read('reports/evidence/poc/run_info.json')
    freeze=read('reports/t07_protocol_freeze.json')
    c=read('reports/submission_context.json') if (ROOT/'reports/submission_context.json').exists() else {}
    name=git('config','--get','user.name');email=git('config','--get','user.email')
    commits=git('log','--format=%H',a.get('audit_baseline_commit',a['remote_main_commit'])+'..HEAD').splitlines()
    lines=['| Run | N ảnh | N vùng che | Tổng (s) | s/100 ảnh | ms/ảnh |','|---|---:|---:|---:|---:|---:|']
    for label,v in runs.items():
        lines.append(f"| {label} | {v['n_images']} | {v['n_regions']} | {v['runtime_seconds']} | {v['runtime_per_100_images_seconds']:.3f} | {v['runtime_seconds']*1000/v['n_images']:.3f} |")
    runtime='\n'.join(lines)
    report=fr'''# Báo cáo thực hành Day 06 — T07 Face/Plate Redaction

## 00 · Header

| Mục | Giá trị |
|---|---|
| Group ID / Challenge | **G07 / T07 — Che mặt, biển số và ưu tiên ảnh cần duyệt** |
| Thành viên và vai trò | **Nguyễn Văn Hùng — 2A202603097**: tích hợp, thuật toán, kiểm thử, review và report. {c.get('group_status','Hồ sơ có một học viên; chưa xác nhận cho phép nộp cá nhân thay nhóm 5–6 người.')} |
| Git URL | [Repo G07](https://github.com/hunghocweb/K5-Day6-L2-NguyenVanHung-2A202603097-DataGovernanceAndPrivacy) |
| Tag `demo-180m` / `final-24h` | Chưa có; `main` ở `{a['remote_main_commit'][:7]}`, local ở `{a['local_head'][:7]}` và có thay đổi chưa commit. |
| Deadline (giờ Việt Nam) | {c.get('deadline','Chưa có ngày/giờ cụ thể; đề quy định demo ở phút180 và hoàn thiện +24 giờ.')} |

## 01 · Problem statement

- **Ai chịu hậu quả khi sai?** Người trong ảnh có thể bị lộ mặt/biển số; người dùng dữ liệu mất tín hiệu khi che nhầm hoặc che quá rộng.
- **Điểm nghẽn:** mặt nhỏ/tối/mờ dễ bị bỏ sót; bbox biển số có thể chưa phủ hết ký tự; kính xe và đầu quay lưng có thể bị che nhầm.
- **Đầu vào → đầu ra:** `images.csv`, ảnh và manifest → `redactions.csv` (`image_id,x,y,w,h`), PNG đã che, queue 15, review, audit và quyết định. [ZIP hiện có](../submissions/T07_G07_review15_candidate.zip).
- **GO / HOLD / REWORK:** GO cần review xong và bằng chứng chất lượng đạt guardrail; HOLD khi thiếu review/đánh giá độc lập; REWORK khi còn reject. Hiện tại **REWORK / REVIEW_REJECTED**, owner **Nguyễn Văn Hùng**.
- **Giả định cần xác nhận:** bài nộp chỉ dùng synthetic; nếu dùng ảnh thật, domain owner/DPO phải xác nhận vùng cần che, quyền sử dụng và phạm vi xuất dữ liệu.

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
- **Trade-offs:** padding tăng coverage nhưng có thể tăng che thừa; rule synthetic dễ nhầm kính; crop đa tỷ lệ tốn thêm lượt dò. [Nhánh synthetic v3](../docs/T07_SYNTHETIC_V3.md) dùng elip/nét mặt và mở rộng chữ hai mép biển số. V2/v3/đa tỷ lệ là thử nghiệm riêng, chưa thay kết quả máy gốc hoặc chứng minh tốt hơn trên ảnh thật.

## 04 · Baseline vs PoC

- **Baseline / version:** YuNet + YOLO mặc định, blur sigma 12, không correction; `src.t07_baseline`.
- **PoC / version:** cùng weights/confidence, thêm rule synthetic, padding 3%, risk queue và sửa vùng; `src.pipeline` → `src.t07_pipeline`. Các run gốc ghi commit `{info['git_sha']}`; [config](../configs/lab.yaml), [freeze](t07_protocol_freeze.json).
- **Cùng điều kiện:** 60 ảnh, 36 dev / 24 eval, seed 42; evaluator dùng 24 eval. Baseline không dùng người sửa theo định nghĩa đề; PoC sau review dùng B=15. Báo riêng PoC máy để tách tác dụng thuật toán và thao tác người. Ba lượt web ngoài queue được giữ riêng, không áp vào bản review15.

**Bảng 1 — Metric chất lượng:** chưa có output `src.compare` do thiếu GT; các ô sau chỉ ghi trạng thái, không phải điểm đã đo.

| Metric | Baseline | PoC máy | Sau review15 |
|---|---|---|---|
| PII Miss Rate ↓ | Chưa đo | Chưa đo | Chưa đo |
| Over-redaction ≤0.30 | Chưa đo | Chưa đo | Chưa đo |
| Hard slices / Recall@B | Chưa đo | Chưa đo | Chưa đo |

**Thống kê vận hành** do script lấy từ run, tách khỏi điểm chất lượng:

{runtime}

Bản sau review15 có **{r['n_regions']} vùng che**, giữ nguyên **{r['unchanged_outside_queue']} PNG ngoài queue**. Đây là xuất lại quyết định đã lưu, không phải inference mới; không lấy thời gian xuất lại để so detector. Blur-all là đối chứng bắt buộc và vẫn phải chấm guardrail. Bbox tăng không chứng minh recall tăng.

## 05 · Evaluation protocol

- **Primary / TP-FN / denominator:** PII Miss Rate = số vùng GT có hợp pixel che phủ <90% diện tích / tổng vùng GT cần che. Đơn vị là vùng; đủ coverage là hit, chưa đủ là miss; không thay bằng IoU 0.5.
- **FP / che thừa:** số pixel che ngoài hợp vùng GT / tổng pixel che, guardrail≤0.30; không đếm FP bằng IoU bbox.
- **Threshold / budget / lúc chốt:** face 0.90, plate 0.25, padding 0.03, blur sigma 12, B=15; freeze trước lần all đầu tại `{freeze['frozen_at']}`. Hash detector/model/config còn khớp; contract và pipeline hiện khác snapshot gốc do bảo trì/bổ sung. Cần freeze bản nộp hiện tại sau commit, không gán hash mới cho thời điểm cũ.
- **Heldout / CI:** tham số đầu chọn trên dev; review sau đó đã xem một số ảnh eval nên không gọi đây là test mới chưa xem. Chấm chính thức cần hidden eval mới của giảng viên. Dự kiến bootstrap theo ảnh 1.000 lần, seed 42, CI 95%; chưa tính khi thiếu nhãn độc lập.
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
- **Latency / compute:** Python {info['machine']['python']}, {info['machine']['platform']}, CPU {info['machine']['processor']}. ms/ảnh ở bảng vận hành gồm xuất PNG/contact sheet, chưa đo riêng inference.

## 07 · Human & decision

- **Queue / B / ưu tiên:** [review_queue.csv](review_queue.csv), B=15 theo risk; reviewer **Nguyễn Văn Hùng**, **{r['review_decisions'].get('approve',0)} approve + {r['review_decisions'].get('edit',0)} edit** trên web; {r['review_splits'].get('dev',0)} dev/{r['review_splits'].get('eval',0)} eval. Dùng timestamp/bbox đã lưu, không tạo lượt duyệt mới khi gom bài.
- **Override / audit:** edit thay toàn bộ danh sách bbox và xuất lại từ ảnh gốc. [Audit web](evidence/audit_web.jsonl) giữ actor, thời gian, before/after, snapshot. Bản sửa đầu có AI hỗ trợ; [journal sửa và xác nhận](evidence/audit_corrections_and_confirmation.jsonl) giữ lịch sử. [Manifest review15](evidence/review15_manifest.json) kiểm 45 ảnh ngoài queue không đổi.
- **Ngoài budget:** pii_00 edit; pii_02 và pii_03 reject. [Ba lượt thêm](evidence/additional_web_reviews.csv) không được tính thành B=18 hoặc áp bbox ngoài queue vào review15. Hai ảnh có [đề xuất sửa](t07_reject_rework.json), bản đầu đã được người dùng yêu cầu làm tiếp; chưa có lượt duyệt đóng lỗi. Bàn duyệt có nút xem đề xuất, không tự ghi quyết định.
- **Policy / reason / owner:** **REWORK / REVIEW_REJECTED / Nguyễn Văn Hùng**; [decision_log.csv](decision_log.csv). Hết reject nhưng thiếu QA độc lập vẫn HOLD; duyệt top-15 chưa đủ tự GO.

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

- **Output pytest:** **{t['passed']} passed, {t['skipped']} skipped, {t['failed']} failed**; JUnit ghi suite {t['seconds']}s. [JSON](t07_current_validation.json) / [XML gốc](evidence/pytest.xml). Skip cần GT/overlay giảng viên; test pass không phải điểm model.
- **Success / failure / edge:** schema/export, reviewer/snapshot/bounds sai, replay audit, pixel ngoài vùng, chặn network/GT, upload lỗi và retry khóa file Windows.
- **Máy / git sha:** máy ở mục 06; run gốc `{info['git_sha']}`, local HEAD `{a['local_head'][:7]}` với thay đổi chưa commit.
- **Web:** `MO_WEB_T07.cmd` để sửa ảnh; `MO_XEM_TUNG_BUOC_T07.cmd` để tải ảnh/xem 6 bước. [QA](t07_walkthrough_validation.json) đã chạy Chrome upload/model thật trên desktop và viewport 390 px.
- **Tái tạo review:** `python -m tools.prepare_t07_submission` cần run/journal local gốc. Người chấm lấy CSV/60PNG trong ZIP; một lệnh inference không tạo lại hành động người chưa nhập. Bản máy và bản sau review được giữ riêng.

## 09 · Limitations & ethics

- **PII / license:** chỉ fixture synthetic được gom nộp. Ảnh upload thật giữ local; không đưa `out/walkthrough`, weights, token hoặc ảnh thật lên repo. Inventory ghi quyền weights YOLO chưa được upstream kết luận đầy đủ.
- **Failure mode upload raw:** gửi ảnh thật lên API/cloud trước blur sẽ lộ dữ liệu trước khi che. Pipeline chạy offline; web nhận ảnh qua loopback và worker dùng weights local, không gọi API ngoài. Đây là rủi ro phòng ngừa, không ghi nhận là một sự cố thực tế đã xảy ra.
- **Điều chưa biết / false alarms:** chưa có metrics chính thức theo GT giảng viên; còn reject; bbox đã duyệt có thể chứa padding hoặc sai sót. [Thử nghiệm giảm che thừa trên dev](t07_precision_dev_experiment.md) đạt 47/49 vùng (95,92%), che thừa 9,35%; còn hai vùng chưa đủ phủ và chẩn đoán bỏ từng ảnh của phần hiệu chỉnh mặt chưa đạt 95%. Giữ REWORK, chưa kết luận giữ được mức này trên ảnh mới.
- **Scale 10×:** chưa benchmark; crop tăng lượt inference, budget 15 cố định giảm tỷ lệ được người xem. Cần đo lại tốc độ và rủi ro ngoài queue.
- **Giả định pháp lý:** blur là redaction, không khẳng định ẩn danh hóa hoàn toàn hoặc đạt GDPR/luật Việt Nam; cần DPO/domain owner xác nhận khi dùng thật.
- **Kế hoạch cải tiến:** đóng hai reject và duyệt lại; bổ sung nhãn dev và ảnh mới có quyền dùng, split theo nguồn/video; chọn rule/ngưỡng trên dev, khóa cấu hình trước khi đo holdout mới. Không tune lại bằng val đã mở. Weights hiện chưa train lại.

## 10 · Team contribution

| Học viên | Git author/email | Module / vai trò | Commit chứng minh | Trình bày? |
|---|---|---|---|---|
| Nguyễn Văn Hùng — 2A202603097 | {name} / {email} | src/t07_*, config, tests, tools, review, report | `{commits[-1][:7]}` code; `{commits[0][:7]}` hồ sơ; web/report mới chưa commit | {c.get('presenter','Chưa có bằng chứng showcase/debate')} |

Đối chiếu `git shortlog -sne --all` với [contributions](team_contributions.csv). Code/đề xuất sửa có Codex hỗ trợ; quyết định web mục 07 do Nguyễn Văn Hùng đã lưu. Chỉ ghi thành viên thực sự có trong hồ sơ.

## 11 · References

| URL / tài liệu | Ngày xem | Ý nghĩa |
|---|---|---|
| [README và template gốc](https://github.com/hunghocweb/K5-Day6-L2-NguyenVanHung-2A202603097-DataGovernanceAndPrivacy/tree/b0ce8bfd972315448aace86185c1a3e075754b93/reports) | 15/11/2026 | Đối chiếu mục00–11 và checklist; snapshot gốc đọc tại `b0ce8bf`. Mốc push đầu của bài T07: `0cd85c3`. |
| [Đề T07](../docs/tracks/T07.md) | 15/11/2026 | Output, coverage 90%, che thừa ≤30%, budget 15, hard negative và failure modes. |
| [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | 14/11/2026, theo inventory | Detector mặt CPU, nguồn và license. |
| [open-image-models](https://github.com/ankandrew/open-image-models) / [issue license](https://github.com/ankandrew/open-image-models/issues/14) | 14/11/2026, theo inventory | Detector plate ONNX và giới hạn xác nhận license weights. |

**Bảng 3 — Checklist chốt khi nộp**

- [x] Config/provenance/seed/model inventory; baseline/PoC/blur-all; hai failure cases có ảnh.
- [x] Queue 15 có reviewer/override/audit/decision; ZIP 60 ảnh, giữ 45 ảnh ngoài queue.
- [x] Offline/tests đạt; README, metric contract và contributions đúng người/commit đã có.
- [ ] Đóng hai reject; chấm baseline/PoC/blur-all, hard slices và bảng so sánh bằng script khi có nhãn hợp lệ.
- [x] Có [5 slide PDF](../demo/T07_G07_slides.pdf) và [HTML offline](../demo/T07_G07_slides.html); [showcase.md](../demo/showcase.md) là kịch bản. Đề không bắt buộc PPTX.
- [ ] Xác nhận nhóm/deadline/showcase; commit bản cuối, freeze/tag đúng mốc, thêm link tag vào README.
- [x] Đã push code T07, report và bằng chứng lên main theo yêu cầu người dùng tại mốc `0cd85c3`; các bổ sung sau đó được ghi bằng commit tiếp theo.
'''
    (ROOT/'reports/group_report.md').write_text(report,encoding='utf-8')
    row={'member':'Nguyễn Văn Hùng (2A202603097)','git_author_name':name,'git_email':email,
         'role':'Integration; algorithm; QA; human web review; report (Codex-assisted code)',
         'modules_files':'src/t07_*;src/web_review;tests/test_t07*;tools/*t07*;configs/lab.yaml;reports',
         'commit_shas':';'.join(reversed(commits)),
         'evidence_note':'Human web decisions: reports/review_queue.csv; source actors retained in audit history',
         'presenter':c.get('presenter','not recorded')}
    (ROOT/'reports/team_contributions.csv').write_bytes(csv_bytes([row],list(row)))
    print('Built sections 00-11 and contributions from actual submission evidence.')


if __name__=='__main__':main()
