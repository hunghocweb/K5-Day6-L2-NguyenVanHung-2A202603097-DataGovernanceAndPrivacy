"""Single routed walkthrough with explicit native and optional user margins."""
import csv
import json
import time
import zipfile
import cv2
from src.t07_auto import AutoDetectors
from src.t07_router import route_image
from src.t07_detectors import clip_box


def run_auto(folder,config,status,finish,annotate,blur):
    begun=time.perf_counter()
    image=cv2.imread(str(folder/'source.png'));height,width=image.shape[:2]
    routing=route_image(image)
    labels={'real':'Ảnh chụp thật','2d':'Ảnh vẽ 2D','both':'Chưa rõ loại ảnh: kiểm tra cả hai nhánh'}
    cv2.imwrite(str(folder/'tiles.png'),image)
    status(2,labels[routing['route']],mode='auto',routing=routing)
    status(3,'Đang tìm mặt và biển số theo loại ảnh.',mode='auto',routing=routing)
    root=folder.parents[2]
    trace=AutoDetectors({'model_inventory':str(root/'reports/t07_models.json')}).detect_details(image,routing=routing)
    raw,merged=trace['raw'],trace['merged']
    cascade=trace.get('cascade');intrinsic=trace.get('detector_padding_applied',False)
    selected=cascade.get('selected') if cascade else None
    branch_label={'ellipse_blur_2d':'2D elip + xử lý mờ bản gốc','detailed_2d':'2D chi tiết dự phòng'}.get(selected,labels[routing['route']])
    cv2.imwrite(str(folder/'detections.png'),annotate(image,raw))
    cv2.imwrite(str(folder/'merged.png'),annotate(image,merged))
    status(4,branch_label+' · đã xác định vùng che.',mode='auto',routing=routing,cascade=cascade,
           detector_padding_applied=intrinsic)
    padded=[clip_box(dict(b),width,height,config['padding']) for b in merged]
    cv2.imwrite(str(folder/'padding.png'),annotate(image,padded))
    status(5,'Đã áp dụng mức nới bổ sung bạn chọn.',mode='auto',routing=routing,cascade=cascade,
           detector_padding_applied=intrinsic)
    cv2.imwrite(str(folder/'result.png'),blur(image,padded))
    with (folder/'redactions.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['image_id','x','y','w','h','type','confidence','source'],extrasaction='ignore')
        writer.writeheader();writer.writerows(dict(image_id=folder.name,**b) for b in padded)
    report=dict(job=folder.name,state='complete',stage=6,message='Hoàn tất xử lý tự động; xem lại ảnh kết quả.',
        started=config['started'],mode='auto',padding=config['padding'],original_size=config['original_size'],
        working_size=[width,height],n_views=len(trace['views']),raw_count=len(raw),merged_count=len(merged),
        final_count=len(padded),n_face=sum(b['type']=='face' for b in padded),
        n_plate=sum(b['type']=='plate' for b in padded),seconds=round(time.perf_counter()-begun,3),boxes=padded,
        offline=True,quality_metrics=None,human_review=False,requires_human_review=True,
        routing=routing,algorithm=trace['algorithm'],review_reasons=trace['review_reasons'],
        cascade=cascade,selected_algorithm=trace.get('selected_algorithm'),
        padding_policy=trace.get('padding_policy'),
        bbox_calibration_applied=False,detector_padding_applied=intrinsic,
        padding_applications=None if intrinsic else 1,user_padding_applications=1,
        stage_summary=branch_label+f" · {sum(b['type']=='face' for b in padded)} mặt · {sum(b['type']=='plate' for b in padded)} biển số · nới bổ sung {config['padding']:.0%}.",
        notice=('Bản 2D gốc giữ mức nới sẵn của thuật toán. ' if intrinsic else '')+
               'Thanh trượt chỉ nới thêm vùng che. Kết quả tự động cần kiểm tra mặt/biển số còn lộ và vùng che nhầm; cổng chọn nhánh chưa được xác thực độc lập.')
    (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with zipfile.ZipFile(folder/'result.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in ['result.png','redactions.csv','report.json']:archive.write(folder/name,name)
    finish(report)
