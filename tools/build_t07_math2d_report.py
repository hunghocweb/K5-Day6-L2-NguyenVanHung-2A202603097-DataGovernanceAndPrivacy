"""Publish the frozen mathematical experiment, including its failed guardrail."""
import json,re,shutil,xml.etree.ElementTree as ET
from datetime import datetime,timezone
from tools.evaluate_t07_reviewed import ROOT,EVIDENCE,sha,write_json


def main():
    read=lambda path:json.loads((ROOT/path).read_text(encoding='utf-8'))
    frozen=read('reports/t07_math2d_freeze.json')
    metrics={split:read(f'out/t07_math2d_{split}/metrics.json') for split in ['dev','eval']}
    once=read('reports/t07_math2d_validation_once.json')
    assert once['state']=='complete'
    assert all(sha(ROOT/path)==value for path,value in frozen['source_sha256'].items())
    assert all(m['source_sha256']==frozen['source_sha256'] for m in metrics.values())
    assert all(m['config']==frozen['config'] for m in metrics.values())
    assert sha(EVIDENCE/'regions_dev.csv')==frozen['dev_reference_sha256']
    assert sha(EVIDENCE/'regions_eval.csv')==metrics['eval']['reference_sha256']
    browser=read('out/t07_walkthrough_qa_math2d/results.json')
    assert all(r['ok'] and r['report']['mode']=='math2d' and r['report']['padding']==0 for r in browser.values())
    suites=ET.parse(ROOT/'out/t07_math2d_tests.xml').getroot().findall('testsuite')
    counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    counts['passed']=counts['tests']-counts['failures']-counts['errors']-counts['skipped']
    evidence=ROOT/'reports/evidence/math2d';evidence.mkdir(exist_ok=True)
    for split in ['dev','eval']:
        for name in ['metrics.json','predictions.json','predictions.csv','coverage.csv','trace.json','machine_audit.jsonl']:
            shutil.copy2(ROOT/f'out/t07_math2d_{split}'/name,evidence/f'{split}_{name}')
    shutil.copy2(ROOT/'reports/t07_math2d_freeze.json',evidence/'freeze.json')
    shutil.copy2(ROOT/'out/t07_math2d_tests.xml',evidence/'pytest.xml')
    for original,target in [('results.json','walkthrough_results.json'),('desktop.png','walkthrough_desktop.png'),('mobile.png','walkthrough_mobile.png')]:
        shutil.copy2(ROOT/'out/t07_walkthrough_qa_math2d'/original,evidence/target)
    validation=dict(source_matches_freeze=True,dev_reference_matches=True,eval_reference_matches=True,
        validation_once_complete=True,full_suite_before_web_integration=counts,
        targeted_web_and_math_stdout=dict(passed=12,failed=0),
        browser={k:dict(ok=r['ok'],checks=len(r['tests'])) for k,r in browser.items()},
        human_audit_sha256=sha(ROOT/'out/web_review/audit_log.jsonl'),
        human_audit_unchanged=sha(ROOT/'out/web_review/audit_log.jsonl')=='d60d1e1fff07581602c6d997852e13998fdde437b58d5c70386de111b8fb5604')
    assert validation['human_audit_unchanged'] and counts['failures']==counts['errors']==0
    coverage_pass=metrics['dev']['region_success_rate']>=.95 and metrics['eval']['region_success_rate']>=.90
    overall=coverage_pass and all(m['guardrail_pass'] for m in metrics.values())
    result=dict(at=datetime.now(timezone.utc).isoformat(),status='PASS' if overall else 'REWORK_OVER_REDACTION',
        coverage_targets_pass=coverage_pass,overall_quality_pass=overall,metrics=metrics,checks=validation,
        reference_kind='user_reviewed_boxes_with_padding',independent_holdout=False,
        source_freeze='reports/t07_math2d_freeze.json',
        warning='Internal reviewed subset only. Val was measured once after dev freeze; no retuning. Instructor GT is unavailable.')
    write_json(ROOT/'reports/t07_math2d_experiment.json',result)
    write_json(ROOT/'reports/t07_math2d_validation.json',validation)
    lines=['# T07 — Thử nghiệm thuật toán toán học trên ảnh 2D','',
        '## 1. Kết quả đối chiếu mục tiêu','',
        '**Chưa đạt yêu cầu tổng thể.** Bản đã khóa đạt mục tiêu độ phủ dev ≥95% và val ≥90%, '
        'nhưng che thừa trên val là50,22%, vượt giới hạn30%. Trạng thái **REWORK_OVER_REDACTION**; '
        'chỉ cung cấp chế độ thử trên web, giữ nguyên cấu hình pipeline mặc định.','',
        '## 2. Dữ liệu và protocol','',
        'Giữ split gốc36dev/24eval và seed42. Chạy từ pixel đủ60ảnh; chỉ9ảnh dev/49vùng và '
        '6ảnh eval/31vùng có bbox người dùng đã duyệt để tính chất lượng. Không coi45ảnh còn lại là negative. '
        'Những nhãn này nằm trong đúng queue15ảnh đã cố định, không tăng ngân sách review.','',
        'Đây là nhãn tham chiếu nội bộ lấy từ vùng che đã duyệt, có padding. Không thay GT giảng viên. '
        'Dữ liệu từng được xem/review ở vòng trước nên val không phải holdout độc lập hoàn toàn. '
        'Trong vòng toán học này chỉ chỉnh trên dev; khóa nguồn/phụ thuộc/config/environment trước khi '
        'đọc val, chạy val một lần và giữ nguyên thuật toán sau kết quả.','',
        '## 3. Thuật toán','',
        '1. Ước lượng Gaussian blur bằng tỷ lệ năng lượng gradient/Laplacian. Tạo các view Wiener '
        'gain0,8/1,0/1,2, regularization0,003; giữ ảnh gốc cho kiểm tra và xuất kết quả.',
        '2. Nhánh mặt thô dùng mask màu và ellipse/nét mặt v3 trên ảnh gốc và các view phục hồi. '
        'Hợp nhất giữ độ phủ; xác nhận lại bằng chứng mắt–miệng trên pixel gốc để loại nét giả do ringing.',
        '3. Nhánh mặt tinh tìm contour Canny/MSER trong các thành phần màu cục bộ, fit ellipse bằng '
        'AMS/moment. Cung đầu bị che được hoàn thiện bằng RANSAC có lọc độ cong và giới hạn ngoại suy. '
        'Chuẩn hóa ellipse, kiểm tra nét tối trên cả luma/chroma và so ellipse trống với ellipse có nét mặt. '
        'Mặt nhỏ v4 có khung tối đa20px bổ sung các trường hợp nhỏ.',
        '4. Nhánh biển số tìm thân xe màu tương đối đồng nhất có bằng chứng cửa kính. Chiếu đối nghịch '
        'màu R−G/B−G để tìm panel trắng/đen gần trung tính ngay cả khi blur hòa chữ vào nền. '
        'Nối các mảnh cùng hàng trong thân xe; ràng buộc tỷ lệ và vị trí để hạn chế dính đèn.',
        '5. Margin biển chọn trên dev: ngang tối thiểu8px, dọc tối thiểu4px, có sàn theo chiều cao nét. '
        'Không mở rộng theo biển thô. Mặt thô nhỏ thêm1px; hợp nhất cùng lớp giữ độ phủ. '
        'Blur sigma12 trên ảnh gốc trong các bbox cuối.',
        '','Suy luận trong `src/t07_math2d.py` chỉ nhận mảng pixel. Không đọc nhãn, tên ảnh, '
        'manifest hay dự đoán lưu sẵn. Đây là cải thiện thuật toán2D; không huấn luyện lại YuNet/YOLO.','',
        '## 4. Kết quả và cách tính','',
        'Một vùng thành công khi hợp mask che phủ ≥90% vùng tham chiếu. Success=1−PII Miss Rate. '
        'Che thừa=pixel che ngoài hợp GT/tổng pixel che. Dùng hàm metric của repo; CI bootstrap '
        'theo ảnh1000lần, seed42. Đây không phải mAP hoặc confidence detector.','',
        '| Tập | Ảnh chạy / ảnh có nhãn | Vùng đủ phủ | Miss | Che thừa | Gate ≤30% |',
        '|---|---:|---:|---:|---:|---|']
    for split,m in metrics.items():
        lines.append(f"| {'dev' if split=='dev' else 'val (eval)'} | {m['n_processed_images']} / {m['n_images']} | "
            f"{m['n_reference_regions']-m['missed_regions']}/{m['n_reference_regions']} ({m['region_success_rate']:.2%}) | "
            f"{m['pii_miss_rate']:.2%} | {m['over_redaction_rate']:.2%} | {'PASS' if m['guardrail_pass'] else 'FAIL'} |")
    lines+=['','| Tập | Mặt đủ phủ | Biển số đủ phủ | CI95% success | CI95% che thừa |','|---|---:|---:|---|---|']
    for split,m in metrics.items():
        f,p=m['by_type']['face'],m['by_type']['plate'];lo,hi=m['bootstrap_95ci_miss'];ol,oh=m['bootstrap_95ci_over']
        lines.append(f"| {split} | {f['n_regions']-f['missed']}/{f['n_regions']} | {p['n_regions']-p['missed']}/{p['n_regions']} | "
            f"{1-hi:.2%}–{1-lo:.2%} | {ol:.2%}–{oh:.2%} |")
    lines+=['','Runtime chạy36dev: '+f"{metrics['dev']['seconds']:.2f}s; chạy24eval: {metrics['eval']['seconds']:.2f}s. "
        'Đây là thời gian inference và xuất bằng chứng máy, không bao gồm review người.','',
        '## 5. Kiểm chứng','',
        '- Toàn repo trước nối web:124passed,39skipped,0failed. Cả39skip do thiếu overlay GT giảng viên.',
        '- Sau nối web:12test web/cascade passed. Bộ ảnh tự tạo kiểm tra đầu có nét/đầu trống, '
        'biển trên xe, blur, thu nhỏ, JPEG và đầu bị che; không dùng crop GT làm template.',
        f"- Chrome thực: desktop{len(browser['desktop']['tests'])} và mobile{len(browser['mobile']['tests'])} kiểm tra pass: "
        'upload/worker, đủ6bước, tải ZIP, mở lại phiên, mode/margin đúng và không tràn ngang. '
        'Ảnh smoke làpii_40thuộc dev; không phải phép đo độ chính xác trên ảnh mới.',
        '- Hash nguồn/phụ thuộc/config của dev và val trùng bản freeze; journal human gốc không đổi. '
        'Audit mới là sự kiện máy, không ghi thành quyết định duyệt của người dùng.','',
        '## 6. Tái lập và chạy thử','',
        'Bấm `MO_XEM_TUNG_BUOC_T07.cmd`, chọn **2D toán học · ellipse và biển số mờ**, tải ảnh và chạy. '
        'Margin bổ sung tự về0% vì cascade đã có margin nội bộ. Với ảnh thật, chọn detector đa tỷ lệ. '
        'Nếu server đang dùng mã cũ, dừng server local rồi mở lại launcher.','',
        '```powershell',
        '.\\.venv\\Scripts\\python.exe -m tools.evaluate_t07_math2d --split dev',
        '.\\.venv\\Scripts\\python.exe -m tools.build_t07_math2d_report',
        '.\\.venv\\Scripts\\python.exe -m tools.smoke_t07_walkthrough --mode math2d','```','',
        'Không chạy lại val: marker `reports/t07_math2d_validation_once.json` đã ghi complete và evaluator '
        'chặn lần đo sau. Kết quả đã lưu đủ để kiểm tra lại phép tính.','',
        '## 7. Giới hạn và công việc còn lại','',
        'Độ phủ cao chưa đủ để đạt lab: val còn vượt ngưỡng che thừa. Hợp nhất bảo thủ và margin '
        'được chọn trên9ảnh dev là giới hạn cần đánh giá thêm; số liệu hiện tại chưa chứng minh tổng quát '
        'sang ảnh thật hoặc mọi kiểu vẽ/blur. Không chọn lại threshold hay sửa hình học dựa trên val này.','',
        'Vòng tiếp theo cần mở rộng nhãn trên phần dev chưa có nhãn, thêm ca âm/ảnh mới có quyền dùng '
        'và chuẩn bị holdout mới trước khi tune tiếp. Chưa đổi quyết định lab thành GO và chưa ghi '
        'người dùng đã duyệt đầu ra mới.','',
        '## 8. Bằng chứng','',
        '- `reports/t07_math2d_experiment.json`: metrics/CI, status và kiểm tra nguồn.',
        '- `reports/t07_math2d_freeze.json`: mã/config/environment khóa trước val.',
        '- `reports/t07_math2d_validation_once.json`: dấu lần đo val duy nhất.',
        '- `reports/evidence/math2d/`: metrics, predictions, coverage, trace, audit máy, JUnit và QA web.',
        '- `reports/evidence/math2d/source/`: snapshot mã được đo.',
        '- `reports/evidence/reviewed_reference/`: nguồn nhãn duyệt và provenance.','']
    # Keep identifiers/commands verbatim, separate quantities from prose units.
    formatted=[];in_code=False
    for line in lines:
        if line.startswith('```'):in_code=not in_code;formatted.append(line);continue
        if not in_code:
            parts=re.split(r'(`[^`]*`)',line)
            for i in range(0,len(parts),2):
                parts[i]=re.sub(r'([A-Za-zÀ-ỹ]{2,})(?=\d)',r'\1 ',parts[i])
                parts[i]=re.sub(r'(?<=\d)(?=[a-zà-ỹ])',' ',parts[i])
            line=''.join(parts)
        formatted.append(line)
    (ROOT/'reports/t07_math2d_experiment.md').write_text('\n'.join(formatted),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],coverage_targets_pass=coverage_pass,
        overall_quality_pass=overall,checks=validation),ensure_ascii=True),flush=True)


if __name__=='__main__':main()
