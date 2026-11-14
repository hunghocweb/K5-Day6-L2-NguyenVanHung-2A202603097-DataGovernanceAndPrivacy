"""Isolated offline worker and local upload/session helpers for the walkthrough."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import time
import uuid
import warnings
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parents[1]
JOBS=ROOT/'out/walkthrough'
MAX_UPLOAD=15*1024*1024


def write_status(folder, value):
    temp=folder/'status.tmp'
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    # Windows can briefly deny replacement while an HTTP reader/AV has it open.
    # Keep the previous valid JSON visible; retry the atomic swap, never truncate it.
    for attempt in range(40):
        try:
            os.replace(temp,folder/'status.json')
            return
        except PermissionError:
            if attempt==39:raise
            time.sleep(.05)


def decode_upload(raw):
    if not raw or len(raw)>MAX_UPLOAD:raise ValueError('Chọn ảnh dưới 15 MB.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format not in {'JPEG','PNG','WEBP'}:raise ValueError('Chỉ nhận JPG, PNG hoặc WebP.')
                if image.width*image.height>24_000_000:raise ValueError('Ảnh tối đa 24 megapixel.')
                image=ImageOps.exif_transpose(image).convert('RGBA')
                background=Image.new('RGBA',image.size,'white')
                image=Image.alpha_composite(background,image).convert('RGB')
                size=image.size
                image.thumbnail((1280,1280),Image.Resampling.LANCZOS)
                return image,size
    except (Image.DecompressionBombWarning,Image.DecompressionBombError,OSError) as exc:
        raise ValueError('Không đọc được ảnh hoặc ảnh quá lớn.') from exc


def folder_for(job):
    if len(job)!=32 or any(c not in '0123456789abcdef' for c in job):raise ValueError('Mã phiên không hợp lệ.')
    folder=JOBS/job
    if not folder.is_dir():raise FileNotFoundError('Không tìm thấy phiên xử lý.')
    return folder


def start(raw, mode='auto', padding=0.):
    if mode not in {'auto','general','synthetic','synthetic_v3','math2d','precision'}:raise ValueError('Chế độ không hợp lệ.')
    if not 0<=padding<=.25:raise ValueError('Padding phải trong 0–25%.')
    image,size=decode_upload(raw)
    JOBS.mkdir(parents=True,exist_ok=True)
    # The HTTP server serializes this check/start, preventing overlapping CPU-heavy jobs.
    for status in JOBS.glob('*/status.json'):
        data=json.loads(status.read_text(encoding='utf-8'))
        if data.get('state')=='running' and time.time()-data.get('started',0)<300:
            raise ValueError('Đang xử lý một ảnh khác. Chờ hoàn tất rồi thử lại.')
    job=uuid.uuid4().hex
    folder=JOBS/job;folder.mkdir()
    image.save(folder/'source.png')
    config={'mode':mode,'padding':padding,'original_size':size,'working_size':image.size,'started':time.time()}
    (folder/'config.json').write_text(json.dumps(config),encoding='utf-8')
    write_status(folder,{'state':'running','stage':1,'message':'Đã nhận ảnh; đang chuẩn bị xử lý.','started':config['started']})
    try:
        with (folder/'worker.log').open('ab') as log:
            process=subprocess.Popen([sys.executable,'-m','tools.t07_walkthrough','--job',job],cwd=ROOT,
                stdout=log,stderr=log,stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        (folder/'worker.pid').write_text(str(process.pid))
    except OSError:
        write_status(folder,{'state':'error','message':'Không khởi động được model. Xem worker.log.'})
        raise
    return {'job':job}


def annotate(image,boxes):
    result=image.copy()
    for index,b in enumerate(boxes):
        x,y,w,h=(int(b[k]) for k in ('x','y','w','h'))
        color=(80,200,80) if b['type']=='face' else (30,165,250)
        cv2.rectangle(result,(x,y),(x+w,y+h),color,2)
        cv2.putText(result,f"{index+1} {b['type']} {b.get('confidence',0):.2f}",
                    (x,max(12,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1,cv2.LINE_AA)
    return result


def run(job):
    folder=folder_for(job)
    config=json.loads((folder/'config.json').read_text())
    begun=time.perf_counter()
    def status(stage,message,**extra):
        write_status(folder,{'state':'running','stage':stage,'message':message,'started':config['started'],**extra})
    try:
        from src.common.offline import enable_offline_guard
        enable_offline_guard()
        if config['mode']=='auto':
            from tools.t07_auto_walk import run_auto
            from src.t07_pipeline import blur
            run_auto(folder,config,status,lambda report:write_status(folder,report),annotate,blur)
            return
        if config['mode'] in {'math2d','precision'}:
            from tools.t07_math2d_walk import run_math
            from src.t07_pipeline import blur
            run_math(folder,config,status,lambda report:write_status(folder,report),annotate,blur)
            return
        from src.t07_multiscale import MultiScaleDetectors,windows
        from src.t07_detectors import clip_box
        from src.t07_pipeline import blur
        image=cv2.imread(str(folder/'source.png'));h,w=image.shape[:2]
        # Show the exact full-view YuNet input plus the exact crop windows used.
        scale=min(320/w,320/h);rw,rh=max(1,round(w*scale)),max(1,round(h*scale))
        letterbox=np.full((320,320,3),114,dtype=np.uint8)
        left,top=(320-rw)//2,(320-rh)//2
        letterbox[top:top+rh,left:left+rw]=cv2.resize(image,(rw,rh))
        cv2.imwrite(str(folder/'letterbox.png'),letterbox)
        tiles=image.copy()
        views=windows(w,h) if max(w,h)>320 else []
        for x,y,vw,vh in views:cv2.rectangle(tiles,(x,y),(x+vw-1,y+vh-1),(70,190,250),1)
        cv2.imwrite(str(folder/'tiles.png'),tiles)
        status(2,'Giữ tỷ lệ ảnh và chia các ô chồng lấn.')
        detector=MultiScaleDetectors({'model_inventory':str(ROOT/'reports/t07_models.json')})
        trace=detector.detect_trace(image,lambda done,total:status(3,f'Đang dò ô {done}/{total}',done=done,total=total))
        raw=trace['raw'];merged=trace['merged']
        cv2.imwrite(str(folder/'detections.png'),annotate(image,raw))
        cv2.imwrite(str(folder/'merged.png'),annotate(image,merged))
        status(4,'Đã quy đổi tọa độ và loại vùng trùng theo lớp.')
        if config['mode'] in {'synthetic','synthetic_v3'}:
            if config['mode']=='synthetic_v3':from src.t07_rules_v3 import detect
            else:from src.t07_rules_v2 import detect
            merged,_=detect(image,merged)
        padded=[clip_box(b,w,h,config['padding']) for b in merged]
        cv2.imwrite(str(folder/'padding.png'),annotate(image,padded))
        status(5,'Đã nới vùng che; đang xuất ảnh blur.')
        result=blur(image,padded)
        cv2.imwrite(str(folder/'result.png'),result)
        with (folder/'redactions.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['image_id','x','y','w','h','type','confidence','source'],extrasaction='ignore')
            writer.writeheader();writer.writerows({'image_id':job,**b} for b in padded)
        report={'job':job,'state':'complete','stage':6,'message':'Hoàn tất xử lý tự động; xem lại ảnh kết quả.',
            'started':config['started'],'mode':config['mode'],'padding':config['padding'],
            'original_size':config['original_size'],'working_size':[w,h],'n_views':len(trace['views']),
            'raw_count':len(raw),'merged_count':len(trace['merged']),'final_count':len(padded),
            'n_face':sum(b['type']=='face' for b in padded),'n_plate':sum(b['type']=='plate' for b in padded),
            'seconds':round(time.perf_counter()-begun,3),'boxes':padded,'offline':True,
            'quality_metrics':None,'human_review':False,
            'notice':'Ảnh xuất theo kích thước làm việc, tối đa 1280 px; chưa chứng nhận che hết PII.'}
        (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        with zipfile.ZipFile(folder/'result.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in ['result.png','redactions.csv','report.json']:z.write(folder/name,name)
        write_status(folder,report)
    except Exception as exc:
        write_status(folder,{'state':'error','stage':0,'message':f'Xử lý thất bại: {type(exc).__name__}: {exc}',
                             'started':config['started']})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True)
    run(parser.parse_args().job)
