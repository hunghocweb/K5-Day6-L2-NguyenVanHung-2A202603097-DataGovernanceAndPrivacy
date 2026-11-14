"""Build a five-slide offline HTML deck and printable PDF from recorded evidence."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path
import subprocess
import tempfile

from tools.t07_review_web import ROOT


def embedded(path):
    mime='image/jpeg' if path.suffix=='.jpg' else 'image/png'
    return 'data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()


def build():
    runs=json.loads((ROOT/'reports/t07_run_summary.json').read_text(encoding='utf-8'))['runs']
    tests=json.loads((ROOT/'reports/t07_current_validation.json').read_text(encoding='utf-8'))
    row=''.join(f"<tr><td>{name}</td><td>{r['n_images']}</td><td>{r['n_regions']}</td><td>{r['runtime_seconds']:.4f}</td></tr>" for name,r in runs.items())
    samples=''
    for key,label in [('pii_10','Ký tự biển số còn lộ ngoài bbox'),('pii_40','Mặt mờ bị sót, kính xe bị che nhầm')]:
        samples+=f'<div class="case"><h3>{key} · {label}</h3><div class="pair">'
        for name,title in [('poc','PoC tự động'),('review15','Sau sửa và review')]:
            samples+=f'<figure><img src="{embedded(ROOT/f"demo/t07_samples/{key}/{name}.png")}" alt="{key} {title}"><figcaption>{title}</figcaption></figure>'
        samples+='</div></div>'
    bodies=[
        '''<div class="cover"><p class="eyebrow">G07 · DAY06 DATA LAB</p><h1>Che mặt và biển số<br>có review và audit</h1><p class="lead">T07 · Automatic Face/Plate Redaction</p><div class="rule"></div><p>Nguyễn Văn Hùng · 2A202603097</p><p class="muted">60 ảnh synthetic · 36 dev / 24 eval · seed 42 · budget review 15 ảnh</p></div>''',
        '''<h2>Pipeline phát hiện, ưu tiên và ghi lại sửa đổi</h2><div class="flow"><div><b>1</b><strong>Load</strong><span>Kiểm manifest<br>và SHA256</span></div><div><b>2</b><strong>Process</strong><span>YuNet + YOLO<br>rule, NMS, padding</span></div><div><b>3</b><strong>Rank</strong><span>Risk score<br>top-15 ảnh</span></div><div><b>4</b><strong>Review</strong><span>Sửa bbox<br>và lưu reviewer</span></div><div><b>5</b><strong>Decide</strong><span>PNG, CSV, audit<br>HOLD / REWORK</span></div></div><div class="columns"><div><h3>Thuật toán PoC</h3><p>Thêm dấu hiệu mặt/biển số synthetic, nới bbox 3% và ưu tiên vùng nhỏ, tối, bất định.</p></div><div><h3>Chạy offline</h3><p>CPU offline; weights kiểm hash, không tự tải mạng. Đầu quay lưng là hard negative của đề.</p></div></div><p class="source">Nguồn: docs/tracks/T07.md · configs/lab.yaml · src/t07_pipeline.py · reports/t07_models.json</p>''',
        f'''<h2>Kết quả chạy baseline, PoC và blur-all</h2><table><thead><tr><th>Bản chạy</th><th>Ảnh</th><th>Vùng che</th><th>Tổng thời gian (s)</th></tr></thead><tbody>{row}</tbody></table><div class="columns"><div><h3>Primary và guardrail</h3><p>PII Miss Rate: coverage &lt;90% theo vùng.<br>Over-redaction ≤30% theo pixel.</p></div><div><h3>Cascade: 9 ảnh dev / 49 vùng</h3><p>47/49 vùng đủ phủ (95,92%).<br>Che thừa 28,55%.</p></div></div><p class="callout">Thời gian ở bảng gồm xử lý, xuất ảnh và contact sheet.</p><p class="source">Nguồn: reports/t07_run_summary.json · evidence/cascade_dev/metrics.json · metric_contract.md</p>''',
        f'''<h2>Hai lỗi đã tìm, sửa và lưu ảnh trước–sau</h2><div class="cases">{samples}</div><p class="source">Nguồn: demo/t07_samples/ · quan sát thị giác trên dev, không phải metric GT độc lập</p>''',
        f'''<h2>Hoàn tất review15 và xuất hồ sơ</h2><div class="columns final"><div><h3>Kết quả</h3><ul><li>15 ảnh: 14 approve + 1 edit của Nguyễn Văn Hùng.</li><li>355 vùng che; 45 ảnh ngoài queue giữ nguyên.</li><li>{tests['passed']} passed, {tests['skipped']} skipped, {tests['failed']} failed.</li><li>Code và hồ sơ đã push main; có hai tag nộp bài.</li></ul></div><div><h3>Demo và đánh giá</h3><ul><li>Đã chuẩn bị 5 slide; chưa trình bày demo.</li><li>Chấm GT để đo baseline, PoC, blur-all và các slice.</li><li>Thực hành: 14:00 ngày14/11 đến 12:00 ngày15/11/2026.</li></ul></div></div><p class="callout">Queue15: 14 approve + 1 edit. Quyết định: HOLD / INDEPENDENT_QA_PENDING.</p><p class="source">Nguồn: reports/review_queue.csv · t07_current_validation.json · decision_log.csv · audit nguồn</p>'''
    ]
    slides=''.join(f'<article class="slide" id="slide-{i+1}">{body}<footer>T07 · G07 <span>{i+1} / 5</span></footer></article>' for i,body in enumerate(bodies))
    css='''*{box-sizing:border-box}html,body{margin:0;background:#1c2923;color:#243a2e;font-family:"Segoe UI",Arial,sans-serif}#frame{position:absolute;left:50%;top:50%;transform-origin:center center;width:1280px;height:720px}.slide{display:none;position:absolute;width:1280px;height:720px;padding:56px 64px 60px;background:#f5f6ef;overflow:hidden}.slide.active{display:block}h1{font-size:62px;line-height:1.12;letter-spacing:-1.8px;margin:20px 0}h2{font-size:38px;line-height:1.2;letter-spacing:-.8px;margin:0 0 30px;max-width:1100px}h3{font-size:23px;margin:0 0 14px}p,li{font-size:24px;line-height:1.5;margin:12px 0}ul{padding-left:25px;margin:0}.eyebrow{letter-spacing:2px;font-size:17px;color:#426550}.lead{font-size:27px}.muted{color:#536653}.cover{padding-top:20px}.rule{width:150px;height:5px;background:#296043;margin:32px 0}.flow{display:grid;grid-template-columns:repeat(5,1fr);gap:22px;margin-top:44px;border-top:2px solid #8ca78f;padding-top:26px}.flow>div{display:flex;flex-direction:column;gap:12px}.flow b{font-size:17px;color:#54705e}.flow strong{font-size:26px}.flow span{font-size:20px;line-height:1.5}.columns{display:grid;grid-template-columns:1fr 1fr;gap:60px;margin-top:44px}.columns p,.columns li{font-size:23px}.final{margin-top:35px;gap:50px}.final li{margin-bottom:16px}.source{position:absolute;bottom:57px;left:64px;right:64px;font-size:15px;line-height:1.4;color:#607060}.callout{padding:18px 22px;background:#e3ebdd;border-left:4px solid #296043;font-size:22px;margin-top:28px}footer{position:absolute;bottom:24px;left:64px;right:64px;font-size:15px;color:#647564}footer span{float:right}table{border-collapse:collapse;width:100%;font-size:24px;margin-top:30px}th,td{padding:19px 16px;text-align:left;border-bottom:1px solid #c5d0bf}th{font-size:19px;color:#526c58}.cases{display:grid;gap:22px}.case h3{font-size:20px;margin-bottom:7px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:26px}.pair figure{margin:0;display:grid;grid-template-columns:360px 1fr;gap:10px;align-items:center}.pair img{width:360px;height:203px;object-fit:contain}.pair figcaption{font-size:19px;line-height:1.5}.cases+.source{bottom:53px}#controls{position:fixed;bottom:15px;left:50%;transform:translateX(-50%);display:flex;gap:12px;align-items:center;color:#fff;font-size:14px}button{font:inherit;background:#f5f6ef;color:#243a2e;border:0;border-radius:5px;padding:9px 15px;cursor:pointer}button:focus-visible{outline:3px solid #73b4e7}#counter{min-width:40px;text-align:center}@media print{@page{size:1280px 720px;margin:0}html,body{background:#fff;width:1280px}#frame{position:static;transform:none!important;width:1280px;height:auto}.slide,.slide.active{display:block;position:relative;break-after:page;page-break-after:always;width:1280px;height:720px;print-color-adjust:exact;-webkit-print-color-adjust:exact}.slide:last-child{break-after:auto;page-break-after:auto}#controls{display:none}}'''
    css+='''.slide:nth-child(3) th,.slide:nth-child(3) td{padding:14px 16px}.slide:nth-child(3) .columns{margin-top:26px}.slide:nth-child(3) .callout{margin-top:18px;font-size:21px}.flow{grid-template-columns:repeat(5,minmax(0,1fr))}.flow>div{min-width:0}'''
    js='''let current=0;const slides=[...document.querySelectorAll('.slide')];function show(n){current=Math.max(0,Math.min(4,n));slides.forEach((s,i)=>s.classList.toggle('active',i===current));document.getElementById('counter').textContent=(current+1)+' / 5';location.hash=String(current+1)}function resize(){document.getElementById('frame').style.transform='translate(-50%,-50%) scale('+Math.min((innerWidth-24)/1280,(innerHeight-90)/720)+')'}document.getElementById('prev').onclick=()=>show(current-1);document.getElementById('next').onclick=()=>show(current+1);document.getElementById('print').onclick=()=>print();document.getElementById('full').onclick=()=>document.documentElement.requestFullscreen?.();addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();show(current+1)}if(['ArrowLeft','PageUp'].includes(e.key))show(current-1)});addEventListener('resize',resize);show((parseInt(location.hash.slice(1))||1)-1);resize();'''
    page='<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>T07 G07 · 5 slide demo</title><style>'+css+'</style><div id="frame">'+slides+'</div><nav id="controls"><button id="prev">← Trước</button><span id="counter"></span><button id="next">Tiếp →</button><button id="full">Toàn màn hình</button><button id="print">In / PDF</button></nav><script>'+js+'</script></html>'
    target=ROOT/'demo/T07_G07_slides.html';target.write_text(page,encoding='utf-8')
    chrome=Path(r'C:\Program Files\Google\Chrome\Application\chrome.exe')
    pdf=ROOT/'demo/T07_G07_slides.pdf'
    with tempfile.TemporaryDirectory(prefix='t07_slides_') as profile:
        subprocess.run([str(chrome),'--headless=new','--disable-gpu','--no-first-run','--disable-extensions',
            '--disable-background-networking','--allow-file-access-from-files','--user-data-dir='+profile,
            '--no-pdf-header-footer','--virtual-time-budget=3000','--print-to-pdf='+str(pdf),target.as_uri()],
            check=True,capture_output=True,timeout=60,creationflags=subprocess.CREATE_NO_WINDOW)
    if not pdf.is_file() or pdf.stat().st_size<10000:raise ValueError('PDF export failed')
    print(json.dumps({'slides':5,'html':str(target.relative_to(ROOT)),'pdf':str(pdf.relative_to(ROOT)),'pdf_bytes':pdf.stat().st_size}))


if __name__=='__main__':build()
