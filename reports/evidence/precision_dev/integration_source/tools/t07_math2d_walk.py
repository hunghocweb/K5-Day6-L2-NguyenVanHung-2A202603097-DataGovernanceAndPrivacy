"""Walkthrough artifacts for the frozen 2D cascade, with no human decisions."""
import csv,json,time,zipfile
import cv2
from src.t07_math2d import detect_details
from src.t07_restoration import restore_for_analysis
from src.t07_detectors import clip_box


def run_math(folder,config,status,finish,annotate,blur):
    begun=time.perf_counter()
    mode=config['mode']
    image=cv2.imread(str(folder/'source.png'));height,width=image.shape[:2]
    restored,quality=restore_for_analysis(image,gain=1.2)
    cv2.imwrite(str(folder/'tiles.png'),restored)
    status(2,'Đã ước lượng blur và tạo ảnh phục hồi để dò vùng.',mode=mode)
    status(3,'Đang tìm ellipse, kiểm tra nét mặt gốc và panel biển số.',mode=mode)
    if mode=='precision':
        from src.t07_precision_pipeline import detect_details as precise_detect
        result=precise_detect(image);stages=result['stages']
        raw,intermediate=stages['proposals'],stages['refined']
        summary=f"{len(raw)} đề xuất ban đầu · {len(intermediate)} vùng qua khớp biên và truy vết chữ."
    else:
        result=detect_details(image);stages=result['stages']
        raw,intermediate=stages['coarse'],stages['fine']+stages['small']
        summary=f"{len(raw)} vùng mặt từ nhiều mức phục hồi · {len(intermediate)} ứng viên hình học/mặt nhỏ."
    cv2.imwrite(str(folder/'detections.png'),annotate(image,raw))
    cv2.imwrite(str(folder/'merged.png'),annotate(image,intermediate))
    status(4,'Đã kiểm tra hình học và nét mặt trên ảnh gốc.',mode=mode)
    padded=[clip_box(b,width,height,config['padding']) for b in result['boxes']]
    cv2.imwrite(str(folder/'padding.png'),annotate(image,padded))
    status(5,'Đã hoàn tất biên vùng che; đang xuất kết quả.',mode=mode)
    cv2.imwrite(str(folder/'result.png'),blur(image,padded))
    with (folder/'redactions.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['image_id','x','y','w','h','type','confidence','source'],extrasaction='ignore')
        writer.writeheader();writer.writerows(dict(image_id=folder.name,**b) for b in padded)
    report=dict(job=folder.name,state='complete',stage=6,message='Hoàn tất xử lý tự động; xem lại ảnh kết quả.',
        started=config['started'],mode=mode,padding=config['padding'],original_size=config['original_size'],
        working_size=[width,height],n_views=4,raw_count=len(raw),merged_count=len(result['boxes']),
        final_count=len(padded),n_face=sum(b['type']=='face' for b in padded),
        n_plate=sum(b['type']=='plate' for b in padded),seconds=round(time.perf_counter()-begun,3),boxes=padded,
        offline=True,quality_metrics=None,human_review=False,requires_human_review=True,
        algorithm=result['algorithm'],algorithm_config=result['config'],restoration=result['restoration'],
        stage_summary=summary+f" {len(padded)} vùng che cuối cùng.",
        notice='Đây là nhánh dành cho ảnh vẽ 2D. Ảnh xuất chưa được người dùng duyệt; cần kiểm tra PII còn lộ và vùng che nhầm.')
    (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with zipfile.ZipFile(folder/'result.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in ['result.png','redactions.csv','report.json']:archive.write(folder/name,name)
    finish(report)
