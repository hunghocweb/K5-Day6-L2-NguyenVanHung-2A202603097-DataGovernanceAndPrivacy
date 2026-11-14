"""Actual upload/model run plus browser stage navigation in an isolated QA session."""
import argparse
import html
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer

from tools.t07_review_web import ROOT,ReviewStore,make_handler

HARNESS='''<!doctype html><meta charset="utf-8"><style>body{margin:0}iframe{border:0;width:100vw;height:1050px}</style><iframe src="/walkthroughRESUME" id="app"></iframe><pre id="result">running</pre><script>
const tests=[];const out=document.getElementById('result');
async function tick(){await fetch('/__tick');}
async function wait(fn){for(let n=0;n<400;n++){if(fn())return;await tick();}throw Error('Timeout: '+document.getElementById('app').contentDocument.getElementById('status').textContent);}
function check(ok,message){if(!ok)throw Error(message);tests.push(message);}
(async()=>{const f=document.getElementById('app');await wait(()=>f.contentDocument?.getElementById('file'));const w=f.contentWindow,d=f.contentDocument,$=id=>d.getElementById(id);
UPLOAD
await wait(()=>!$('summary').hidden);check($('status').textContent.includes('Hoàn tất'),'real-worker-completed');
check(d.querySelectorAll('#steps button:enabled').length===6,'six-stages-available');
for(let i=0;i<6;i++){d.querySelectorAll('#steps button')[i].click();await wait(()=>$('stageImage').complete&&$('stageImage').naturalWidth>0);check($('stageTitle').textContent.startsWith((i+1)+'.'),'stage-'+(i+1));}
const response=await fetch($('downloadZip').href);check(response.ok&&response.headers.get('Content-Type')==='application/zip','download-zip');
check(d.documentElement.scrollWidth<=w.innerWidth,'no-horizontal-overflow');
const job=new URL(w.location.href).searchParams.get('job');const report=await(await fetch('/api/walk/status?job='+job)).json();
check(!$('algorithm')&&report.mode==='auto','one-automatic-flow');
check(report.bbox_calibration_applied===false&&report.user_padding_applications===1,'no-calibration-single-user-padding');
check(report.detector_padding_applied?report.padding_applications===null:report.padding_applications===1,'native-padding-reported-honestly');
check($('route').textContent.includes('2D')&&report.routing.route==='2d','content-route-shown');
check(d.querySelectorAll('#steps button')[4].textContent.includes('Nới vùng che'),'padding-stage');
check(Number($('pad').value)===Math.round(report.padding*100),'controls-match-job-padding');
out.textContent=JSON.stringify({ok:true,tests,job,report,width:w.innerWidth});
})().catch(error=>out.textContent=JSON.stringify({ok:false,error:error.stack,tests}));</script>'''

UPLOAD='''await wait(()=>$('status').textContent.includes('Chọn ảnh'));const blob=await(await fetch('/__sample')).blob(),transfer=new w.DataTransfer();transfer.items.add(new w.File([blob],'sample.png',{type:'image/png'}));$('file').files=transfer.files;$('file').dispatchEvent(new w.Event('change'));check(!$('run').disabled,'file-selection-enables-run');$('run').click();'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['auto'],default='auto')
    args=parser.parse_args()
    output=ROOT/('out/t07_walkthrough_qa'+('_'+args.mode if args.mode!='general' else ''));output.mkdir(exist_ok=True)
    session=output/('session_'+os.urandom(4).hex())
    store=ReviewStore(output=session);parent=make_handler(store)
    resume={'job':None}
    class Handler(parent):
        def log_message(self,*args):pass
        def do_GET(self):
            if self.path=='/__tick':
                time.sleep(.1);return self.respond(200,b'{}')
            if self.path=='/__sample':
                return self.respond(200,store.source_image('pii_40').read_bytes(),'image/jpeg')
            if self.path in {'/__qa','/__qa_mobile'}:
                mobile=self.path.endswith('mobile')
                page=HARNESS.replace('RESUME','?job='+resume['job'] if mobile else '').replace('UPLOAD','' if mobile else UPLOAD)
                if mobile:page=page.replace('width:100vw;height:1050px','width:390px;height:1550px')
                return self.respond(200,page.encode(),'text/html; charset=utf-8')
            return super().do_GET()
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    results={}
    try:
        for name,size,route in [('desktop','1680,1500','/__qa'),('mobile','390,1600','/__qa_mobile')]:
            with tempfile.TemporaryDirectory(prefix='t07_walk_chrome_') as profile:
                command=[r'C:\Program Files\Google\Chrome\Application\chrome.exe','--headless=new','--disable-gpu',
                    '--no-first-run','--disable-background-networking','--disable-extensions','--hide-scrollbars',
                    '--user-data-dir='+profile,'--window-size='+size,'--force-device-scale-factor=1',
                    '--virtual-time-budget=180000','--dump-dom','--screenshot='+str(output/(name+'.png')),
                    f'http://127.0.0.1:{server.server_port}'+route]
                process=subprocess.run(command,capture_output=True,timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
                dom=process.stdout.decode('utf-8',errors='replace')
                match=re.search(r'<pre id="result">(.*?)</pre>',dom,re.S)
                result=json.loads(html.unescape(match[1])) if match else {'ok':False,'error':'No browser result'}
                results[name]=result
                if not result.get('ok'):
                    (output/(name+'.html')).write_text(dom,encoding='utf-8')
                    raise RuntimeError(str(result))
                resume['job']=result['job']
        (output/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({name:{'ok':value['ok'],'checks':len(value['tests']),'job':value['job'],
                                 'seconds':value['report']['seconds']} for name,value in results.items()}))
    finally:
        server.shutdown();server.server_close()


if __name__=='__main__':main()
