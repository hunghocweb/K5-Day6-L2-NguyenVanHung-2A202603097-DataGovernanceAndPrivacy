"""Build a dev-only result report, validating the frozen source and tests."""
import json,shutil,xml.etree.ElementTree as ET
from datetime import datetime,timezone
from tools.evaluate_t07_reviewed import ROOT,EVIDENCE,write_json,sha


def main():
    read=lambda p:json.loads((ROOT/p).read_text(encoding='utf-8'))
    frozen=read('reports/t07_precision_dev_freeze.json');metrics=read('out/t07_precision_pixel_dev/metrics.json')
    previous=read('out/t07_math2d_dev/metrics.json');diagnostic=read('out/t07_precision_combined_dev/head_cv_diagnostic_metrics.json')
    head_training=read('out/t07_precision_head_calibration_dev/model_2.json')
    browser=read('out/t07_walkthrough_qa_precision/results.json')
    assert metrics['n_processed_images']==36 and metrics['target_pass']
    assert metrics['source_sha256']==frozen['source_sha256']
    assert all(sha(ROOT/p)==h for p,h in frozen['source_sha256'].items())
    assert metrics['reference_sha256']==sha(EVIDENCE/'regions_dev.csv')
    suites=ET.parse(ROOT/'out/t07_precision_full_tests.xml').getroot().findall('testsuite')
    tests={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    tests['passed']=tests['tests']-tests['failures']-tests['errors']-tests['skipped']
    assert tests['failures']==tests['errors']==0
    assert all(v['ok'] and v['report']['mode']=='precision' and v['report']['padding']==0 for v in browser.values())
    human_hash=sha(ROOT/'out/web_review/audit_log.jsonl')
    assert human_hash=='d60d1e1fff07581602c6d997852e13998fdde437b58d5c70386de111b8fb5604'
    archive=ROOT/'reports/evidence/precision_dev';archive.mkdir(exist_ok=True)
    for name in ['metrics.json','predictions.json','predictions.csv','coverage.csv','trace.json','machine_audit.jsonl']:
        shutil.copy2(ROOT/'out/t07_precision_pixel_dev'/name,archive/name)
    shutil.copy2(EVIDENCE/'regions_dev.csv',archive/'reference_dev.csv')
    shutil.copy2(ROOT/'out/t07_precision_full_tests.xml',archive/'pytest.xml')
    shutil.copy2(ROOT/'reports/t07_precision_dev_freeze.json',archive/'freeze.json')
    for src,dest in [('results.json','web_results.json'),('desktop.png','web_desktop.png'),('mobile.png','web_mobile.png')]:
        shutil.copy2(ROOT/'out/t07_walkthrough_qa_precision'/src,archive/dest)
    for path in (ROOT/'out/t07_precision_visual_dev').glob('*.png'):shutil.copy2(path,archive/path.name)
    integration_files=['src/t07_pipeline.py','configs/lab_precision_dev.yaml','tools/t07_walkthrough.py',
        'tools/t07_math2d_walk.py','src/web_review/walkthrough.js','src/web_review/walkthrough.html','MO_PRECISION_T07.cmd']
    integration_hashes={p:sha(ROOT/p) for p in integration_files}
    for path in integration_files:
        destination=archive/'integration_source'/path;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/path,destination)
    for name in ['head_cv_diagnostic_metrics.json','head_cv_diagnostic_coverage.csv','head_cv_diagnostic_predictions.json']:
        shutil.copy2(ROOT/'out/t07_precision_combined_dev'/name,archive/name)
    validation=dict(source_matches_freeze=True,reference_matches=True,dev_only=True,
        tests=tests,browser={k:dict(ok=v['ok'],checks=len(v['tests'])) for k,v in browser.items()},
        human_audit_unchanged=True,human_audit_sha256=human_hash,quality_target_pass=metrics['target_pass'],
        integration_source_sha256=integration_hashes)
    result=dict(at=datetime.now(timezone.utc).isoformat(),status='DEV_TARGET_PASS_REVIEW_REQUIRED',
        metrics=metrics,previous_dev=previous,head_fold_diagnostic=diagnostic,head_training=head_training,
        checks=validation,unseen_image_generalization_proven=False,official_lab_go=False)
    write_json(ROOT/'reports/t07_precision_dev_experiment.json',result)
    write_json(ROOT/'reports/t07_precision_dev_validation.json',validation)
    lo,hi=metrics['bootstrap_95ci_over'];ml,mh=metrics['bootstrap_95ci_miss']
    lines=['# T07 — Cải thiện thuật toán 2D: giảm che thừa dưới 10% trên dev','',
        '## 1. Kết quả đối chiếu mục tiêu','',
        f"Bản `precision-dev-1` đã chạy trực tiếp từ pixel đủ **36 ảnh dev**. Trên **9 ảnh/49 vùng có nhãn đã duyệt**, "
        f"**47/49 vùng (95,92%)** đủ phủ và che thừa **{metrics['over_redaction_rate']:.2%}**, đạt mục tiêu dev ≥95% và <10%. "
        'Trạng thái **DEV_TARGET_PASS_REVIEW_REQUIRED**; vẫn còn hai vùng chưa đạt coverage90%, chưa tự chuyển quyết định lab thành GO.','',
        '| Cấu hình | Vùng đủ phủ | Mặt đủ phủ | Biển đủ phủ | Che thừa |',
        '|---|---:|---:|---:|---:|',
        f"| `math2d-dev-1` | 47/49 | 26/26 | 21/23 | {previous['over_redaction_rate']:.2%} |",
        f"| precision-dev-1 | 47/49 | 25/26 | 22/23 | {metrics['over_redaction_rate']:.2%} |",'',
        f"Che thừa giảm **{100*(previous['over_redaction_rate']-metrics['over_redaction_rate']):.2f} điểm phần trăm**. "
        'Tổng số vùng đủ phủ giữ nguyên; có một mặt giảm coverage và một biển được cải thiện, nên cần xem theo từng loại thay vì chỉ nhìn số tổng.','',
        '## 2. Dữ liệu và phép đo','',
        'Chỉ dùng ảnh, dự đoán và nhãn dev. Bbox tham chiếu lấy từ quyết định người dùng đã duyệt trong queue15 cố định; '
        'không thêm lượt duyệt, không coi 27 ảnh dev chưa có nhãn là negative, không thay nhãn hoặc công thức metric. '
        'Đây là nhãn nội bộ có padding, không phải GT giảng viên.','',
        'Một vùng thành công nếu hợp mask che phủ ít nhất90% vùng tham chiếu. Che thừa=pixel che ngoài hợp GT/tổng pixel che. '
        'Bản cuối dùng metric repo; phép nén tọa độ dùng khi tìm tham số đã được kiểm tra cho kết quả đúng bằng raster pixel, '
        'gồm clipping, tọa độ lẻ và vùng chồng nhau.','',
        f"Số đếm: {metrics['over_outside_pixels']} pixel ngoài GT / {metrics['redacted_pixels']} pixel che. "
        f"CI95% theo bootstrap ảnh1000lần, seed42: success {1-mh:.2%}–{1-ml:.2%}; che thừa {lo:.2%}–{hi:.2%}. "
        'CI che thừa còn vượt10% ở cận trên; điểm trên9ảnh không phải bảo đảm thống kê cho mọi ảnh mới.','',
        '## 3. Thuật toán được chọn','',
        '1. Dùng cascade toán học trước đó tạo đề xuất, giữ ảnh gốc để kiểm chứng. Các view Wiener chỉ hỗ trợ phát hiện.',
        '2. Khớp lại contour ellipse từ mức trộn màu trên ảnh gốc; margin mặt phụ thuộc blur. Mặt nhỏ loại padding dư, '
        'bắt buộc mô hình nét mắt và miệng cùng có hệ số dương, đủ giải thích phần tín hiệu còn lại sau khi loại nền ellipse trống.',
        '3. Hiệu chỉnh biên mặt bằng **một ma trận hệ số chung**: z=(f−mean)/scale, delta=z×beta; '
        'delta điều chỉnh tâm và kích thước theo chiều cao khung. Sáu đặc trưng gồm bias, nghịch đảo chiều cao, blur/chiều cao, '
        'tỷ lệ khung, luminance và độ tương phản cục bộ. Ridge alpha30 được fit trên22vùng/8ảnhdev; '
        'không có ảnh mẫu, image_id, tọa độ GT hoặc bảng tra từng ảnh trong model.',
        '4. Biển số lấy nền gần trung tính trong thân xe. Cửa kính phải có hai cạnh nằm thực sự phía trên thân, '
        'cách halo blur của chính ứng viên. Nét chữ dùng cực tiểu dọc hai phía ở nhiều khoảng cách: '
        'S=min(G(y−d)−G(y),G(y+d)−G(y)); tích lũy theo cột, nối dải chữ vào nền biển và loại đèn có lõi sáng bão hòa. '
        'Dải chữ dài đi qua nền biển được giữ lại ngay cả khi blur nối chữ thành một khối.',
        '5. Margin biển không cộng hai lần và được tách bốn phía: ảnh sắc dùng trái2/phải5/trên1/dưới1px; '
        'ảnh blur dùng trái0/phải3/trên3/dưới2px. Khi biển rất nhỏ bị che chỉ còn một mảnh, '
        'che bảo thủ vùng xe nhỏ có giới hạn và đánh dấu cần review.',
        '6. Blur sigma12 trên ảnh gốc trong bbox cuối. Xuất tọa độ, trace máy và ảnh; không tạo quyết định human.','',
        'Bản chọn không dùng bộ hồi quy biên biển số: thử nghiệm đó có điểm fit tốt nhưng không ổn định khi bỏ từng ảnh. '
        'Cũng không dùng voting tọa độ làm quyết định cuối; biên lớn từ bước đề xuất được thay bằng biên đã khớp và hiệu chỉnh. '
        'Không huấn luyện lại YuNet/YOLO.','',
        '## 4. Kiểm tra độ ổn định và giới hạn','',
        f"Bỏ lần lượt từng ảnh khỏi phần học hệ số mặt, giữ nhánh biển và các tham số đã chọn: "
        f"**46/49 (93,88%)**, che thừa **{diagnostic['over_redaction_rate']:.2%}**. Đây chỉ là chẩn đoán mô-đun mặt trên dev; "
        'không phải kiểm định độc lập cả pipeline và chưa đạt mốc95% về độ phủ. Vì vậy **chưa chứng minh đã tránh được overfit**.','',
        '- Hai vùng còn thiếu coverage90%: mặt trong `pii_10` và biển trong `pii_38`. Có crop nguồn/kết quả/tham chiếu trong evidence để xem trực tiếp.',
        '- Mẫu nhãn nhỏ, lấy từ queue rủi ro và có padding do người duyệt; kết quả không đại diện toàn bộ36ảnh hoặc ảnh thật.',
        '- Các ca hình tự tạo khác chỉ kiểm coverage vật thể, blur/scale/JPEG và loại đầu trống; không dùng chúng để tuyên bố che thừa<10% trên dữ liệu mới.',
        '- Cần bổ sung nhãn dev có nguồn độc lập và kiểm tra thêm trước khi dùng làm kết luận tổng quát. Chế độ web vẫn là bản thử.','',
        '## 5. Kiểm thử và tái lập','',
        f"Toàn repo: **{tests['passed']} passed, {tests['skipped']} skipped, 0 failed**. "
        'Các skip do thiếu overlay GT giảng viên. Hai mươi mốt kiểm thử mới gồm hình học, bằng chứng mặt nhỏ/cửa kính, '
        'không nối nền tối, metric nén đúng pixel, ba ca toàn cascade có blur/scale/JPEG và tích hợp pipeline '
        'không khởi tạo detector ảnh thật, không nới bbox thêm hoặc tạo sự kiện human.','',
        f"Chrome thực: desktop{len(browser['desktop']['tests'])} và mobile{len(browser['mobile']['tests'])} kiểm tra pass; "
        'upload, đủ6bước, ZIP và mở lại đúng mode/margin. Đã xem ảnh QA; ảnh smoke thuộc dev.','',
        '```powershell',
        '.\\.venv\\Scripts\\python.exe -m tools.evaluate_t07_precision_dev',
        '.\\.venv\\Scripts\\python.exe -m tools.render_t07_precision_dev',
        '.\\.venv\\Scripts\\python.exe -m tools.build_t07_precision_dev_report',
        '.\\.venv\\Scripts\\python.exe -m tools.smoke_t07_walkthrough --mode precision','```','',
        'Profile one-command riêng cho dev đã nối `synthetic_algorithm: precision` vào pipeline T07. '
        'Lệnh dưới tạo output, queue và quyết định theo contract; không đổi profile mặc định hoặc tái gán '
        'quyết định duyệt cũ cho snapshot mới:','',
        '```powershell',
        '.\\.venv\\Scripts\\python.exe -m src.main --config configs/lab_precision_dev.yaml --offline --seed 42','```','',
        f"Runtime chạy36dev của lần này: {metrics['seconds']:.2f}s. Các kiểm tra khác chạy đồng thời nên chưa dùng để kết luận tốc độ tăng/giảm. "
        'Freeze gồm nguồn/config/environment và hash nhãn dev; journal human gốc giữ nguyên.','',
        '## 6. Chạy thử và bằng chứng','',
        'Bấm `MO_PRECISION_T07.cmd` để mở trang cổng8767, chọn **2D precision · giảm che dư**. Margin bổ sung tự về0% để giữ đúng cấu hình đã đo; '
        'nới thêm sẽ thay kết quả. Các bước hiển thị đề xuất → khớp biên/nét chữ → hiệu chỉnh → blur.','',
        '- `reports/t07_precision_dev_experiment.json`: metrics, so sánh dev và chẩn đoán bỏ từng ảnh.',
        '- `reports/t07_precision_dev_freeze.json`: snapshot khóa nguồn/config.',
        '- `reports/evidence/precision_dev/`: predictions, reference_dev, coverage, trace, audit máy, JUnit, ảnh kết quả và QA web.',
        '- `configs/t07_precision.json`: một bộ hệ số/tham số chung, không chứa bbox theo từng ảnh.',
        '- Thay đổi vẫn local; chưa thay cấu hình mặc định hoặc ghi người dùng đã duyệt đầu ra mới.','']
    # Format prose quantities without touching commands or inline identifiers.
    import re
    formatted=[];code=False
    for line in lines:
        if line.startswith('```'):code=not code;formatted.append(line);continue
        if not code:
            chunks=re.split(r'(`[^`]*`)',line)
            for i in range(0,len(chunks),2):
                chunks[i]=re.sub(r'([A-Za-zÀ-ỹ]{2,})(?=\d)',r'\1 ',chunks[i])
                chunks[i]=re.sub(r'(?<=\d)(?=[a-zà-ỹ])',' ',chunks[i])
            line=''.join(chunks)
        formatted.append(line)
    (ROOT/'reports/t07_precision_dev_experiment.md').write_text('\n'.join(formatted),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],source_matches=True,dev_target_pass=metrics['target_pass'],tests=tests)),flush=True)


if __name__=='__main__':main()
