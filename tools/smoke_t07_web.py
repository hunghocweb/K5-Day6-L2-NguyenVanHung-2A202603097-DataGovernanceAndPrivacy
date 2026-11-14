"""Chrome headless interaction checks in an isolated local review session."""
import argparse
import json
import os
import subprocess
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

from tools.t07_review_web import ROOT, ReviewStore, make_handler

HARNESS = r'''<!doctype html><meta charset="utf-8"><style>body{margin:0}iframe{width:100vw;height:100vh;border:0}#result{position:fixed;bottom:0;left:0;background:white;color:black;z-index:100;font:12px monospace}</style><iframe src="/"></iframe><pre id="result">RUNNING</pre><script>
const frame=document.querySelector('iframe'),out=document.querySelector('#result');
const wait=async(fn)=>{for(let n=0;n<1000;n++){if(fn())return;await new Promise(r=>setTimeout(r,40));}throw Error('Timeout');};
const check=(condition,message)=>{if(!condition)throw Error(message);};
const tests=[];
(async()=>{await wait(()=>frame.contentDocument?.querySelectorAll('.image-item').length===15);
const w=frame.contentWindow,d=frame.contentDocument,$=id=>d.getElementById(id);
check(d.documentElement.scrollWidth<=w.innerWidth,'Horizontal overflow');tests.push('layout-no-overflow');
$('reviewer').value='';$('decision').value='';$('save').click();check($('saveStatus').classList.contains('error'),'Missing reviewer must block');tests.push('reviewer-required');
d.querySelector('[data-mode="baseline"]').click();await wait(()=>$('preview').getAttribute('src')?.includes('mode=baseline')&&$('preview').naturalWidth>0);tests.push('baseline-image');
d.querySelector('[data-mode="edit"]').click();await wait(()=>$('preview').getAttribute('src')?.startsWith('blob:'));tests.push('live-blur-preview');
const before=$('regionList').children.length;
$('faceTool').click();const canvas=$('editor'),r=canvas.getBoundingClientRect();
const pointer=(type,x,y)=>canvas.dispatchEvent(new w.PointerEvent(type,{pointerId:42,clientX:r.left+x*r.width/640,clientY:r.top+y*r.height/360,bubbles:true}));
// Browser-dispatched synthetic events have no active pointer to capture.
canvas.setPointerCapture=()=>{};
pointer('pointerdown',50,50);pointer('pointermove',90,80);pointer('pointerup',90,80);
check($('regionList').children.length===before+1&&$('boxW').value==='40'&&$('boxH').value==='30','Draw region');
$('selectTool').click();pointer('pointerdown',60,60);pointer('pointermove',70,70);pointer('pointerup',70,70);
check($('boxX').value==='60'&&$('boxY').value==='60','Move region');
pointer('pointerdown',100,90);pointer('pointermove',110,100);pointer('pointerup',110,100);
check($('boxW').value==='50'&&$('boxH').value==='40','Resize region');
$('deleteBox').click();check($('regionList').children.length===before,'Delete region');tests.push('draw-move-resize-delete');
$('addBox').click();check($('regionList').children.length===before+1,'Add region');
$('boxX').value='100';$('boxX').dispatchEvent(new w.Event('change'));check($('boxX').value==='100','Coordinate editor');
$('undo').click();$('regionList').lastElementChild.click();check($('boxX').value!=='100','Undo');
$('deleteBox').click(); // no selection after undo, intentionally harmless
tests.push('add-coordinate-undo');
$('reviewer').value='Web regression test';$('decision').value='edit';$('note').value='Isolated browser smoke test';
$('save').click();await wait(()=>!$('workspace').inert&&$('saveStatus').textContent.startsWith('Đã lưu:'));tests.push('save-review');
const s=await (await fetch('/api/state')).json(),id=$('imageId').textContent,item=s.items.find(i=>i.id===id);
check(item.reviewer==='Web regression test'&&item.decision==='edit','Persisted reviewer');
check(s.gate.completed===1,'Progress');tests.push('persisted-audit-progress');
$('scope').value='all';$('scope').dispatchEvent(new w.Event('input'));check(d.querySelectorAll('.image-item').length===60,'All images');
$('scope').value='eval';$('scope').dispatchEvent(new w.Event('input'));check(d.querySelectorAll('.image-item').length===24,'Eval split');tests.push('split-filters');
$('scope').value='queue';$('scope').dispatchEvent(new w.Event('input'));
await wait(()=>$('preview').complete&&$('preview').naturalWidth>0);
out.textContent=JSON.stringify({ok:true,tests,width:w.innerWidth,overflow:d.documentElement.scrollWidth>w.innerWidth});
})().catch(e=>out.textContent=JSON.stringify({ok:false,error:e.stack,tests}));
</script>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chrome", default=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    args = parser.parse_args()
    output = ROOT / "out/t07_web_qa"
    output.mkdir(exist_ok=True)
    # Unique session so repeated QA cannot overwrite a prior saved review.
    session = output / ("session_" + os.urandom(4).hex())
    store = ReviewStore(output=session)
    parent = make_handler(store)

    class Handler(parent):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path in {"/__qa", "/__qa_mobile"}:
                html = HARNESS
                if self.path == "/__qa_mobile":
                    html = html.replace('width:100vw', 'width:390px')
                return self.respond(200, html.encode(), "text/html; charset=utf-8")
            return super().do_GET()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    results = {}
    try:
        for name, size, route in (("desktop", "1680,1100", "/__qa"), ("mobile", "390,1000", "/__qa_mobile")):
            with tempfile.TemporaryDirectory(prefix="t07_chrome_") as profile:
                command = [args.chrome, "--headless=new", "--disable-gpu", "--no-first-run",
                           "--disable-background-networking", "--disable-extensions", "--hide-scrollbars",
                           f"--user-data-dir={profile}", f"--window-size={size}", "--force-device-scale-factor=1",
                           "--virtual-time-budget=15000", "--dump-dom", f"--screenshot={output / (name+'.png')}",
                           f"http://127.0.0.1:{server.server_port}{route}"]
                result = subprocess.run(command, capture_output=True, timeout=60,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                text = result.stdout.decode("utf-8", errors="replace")
                (output / (name + ".html")).write_text(text, encoding="utf-8")
                import html
                import re
                match = re.search(r'<pre id="result">(.*?)</pre>', text, re.S)
                results[name] = json.loads(html.unescape(match.group(1))) if match else {"ok": False, "error": "No browser result"}
        results["test_session"] = str(session.relative_to(ROOT))
        (output / "result.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(results, ensure_ascii=False, indent=2))
        if not all(results[name].get("ok") for name in ("desktop", "mobile")):
            raise SystemExit(1)
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
