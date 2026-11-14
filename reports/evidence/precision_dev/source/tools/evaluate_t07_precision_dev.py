"""Dev-only, from-pixel verification of the precision candidate. No eval option."""
import argparse,json,time,platform
from datetime import datetime,timezone
import cv2,numpy as np,pandas as pd
from src.t07_precision_pipeline import detect_details
from tools.evaluate_t07_reviewed import ROOT,FIXTURE,EVIDENCE,read_rows,score,write_json,sha

SOURCES=['configs/t07_precision.json','src/t07_precision_pipeline.py','src/t07_precision_heads.py',
    'src/t07_precision.py','src/t07_face_evidence.py','src/t07_head_calibration.py','src/t07_text_extent.py',
    'src/t07_math2d.py','src/t07_geometry.py','src/t07_local_shapes.py','src/t07_fusion.py','src/t07_restoration.py',
    'src/t07_rules_v3.py','src/t07_rules_v4.py','src/t07_router.py','src/t07_detectors.py','src/metrics/detection.py',
    'tools/evaluate_t07_reviewed.py','tools/evaluate_t07_precision_dev.py']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reviewed-only',action='store_true');parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    if args.freeze and args.reviewed_only:raise ValueError('Freeze requires full dev inference')
    hashes={p:sha(ROOT/p) for p in SOURCES}
    ref=pd.read_csv(EVIDENCE/'regions_dev.csv');keys=list(ref.image_id.unique())
    rows={r['image_id']:r for r in read_rows(FIXTURE/'images.csv') if r['split']=='dev'}
    if args.reviewed_only:rows={k:r for k,r in rows.items() if k in keys}
    out=ROOT/'out/t07_precision_pixel_dev';out.mkdir(exist_ok=True)
    predictions={};trace={};started=time.perf_counter()
    for index,(key,row) in enumerate(rows.items()):
        image=cv2.imread(str(FIXTURE/row['path']))
        if image is None or image.shape[:2]!=(int(row['height']),int(row['width'])):raise ValueError('Invalid input image')
        result=detect_details(image);predictions[key]=result['boxes']
        trace[key]={k:v for k,v in result.items() if k not in ['boxes','stages','config']}
        trace[key]['image_sha256']=sha(FIXTURE/row['path'])
        trace[key]['rejected_heads']=result['stages']['rejected_heads'];trace[key]['rejected_plates']=result['stages']['rejected_plates']
        write_json(out/'progress.json',dict(processed=index+1,total=len(rows),image_id=key))
        print(f'dev {index+1}/{len(rows)} {key}',flush=True)
    if {p:sha(ROOT/p) for p in SOURCES}!=hashes:raise RuntimeError('Source changed during inference')
    metrics,coverage=score(ref,predictions,keys,{k:(360,640) for k in keys})
    metrics.update(precision_guard_pass=metrics['over_redaction_rate']<.10,
        target_pass=metrics['region_success_rate']>=.95 and metrics['over_redaction_rate']<.10,
        seed=42,n_processed_images=len(rows),seconds=time.perf_counter()-started,source_sha256=hashes,
        environment=dict(python=platform.python_version(),opencv=cv2.__version__,numpy=np.__version__,pandas=pd.__version__),
        reference_sha256=sha(EVIDENCE/'regions_dev.csv'),reference_kind='user_reviewed_internal_boxes',independent_test=False)
    write_json(out/'metrics.json',metrics);write_json(out/'predictions.json',predictions);write_json(out/'trace.json',trace)
    coverage.to_csv(out/'coverage.csv',index=False)
    pd.DataFrame([dict(image_id=k,**b) for k,boxes in predictions.items() for b in boxes]).to_csv(out/'predictions.csv',index=False)
    with (out/'machine_audit.jsonl').open('w',encoding='utf-8') as handle:
        for key in rows:handle.write(json.dumps(dict(event='machine_detection',actor_kind='machine',image_id=key,
            route='2d',algorithm='precision-dev-1',source_sha256=hashes,image_sha256=trace[key]['image_sha256'],
            requires_human_review=True,human_review_recorded=False,boxes=predictions[key]),ensure_ascii=False)+'\n')
    if args.freeze:
        if not metrics['target_pass']:raise ValueError('Dev target >=95% coverage and <10% excess not met')
        write_json(ROOT/'reports/t07_precision_dev_freeze.json',dict(at=datetime.now(timezone.utc).isoformat(),source_sha256=hashes,
            metrics=metrics,config=json.loads((ROOT/'configs/t07_precision.json').read_text()),development_only=True))
        archive=ROOT/'reports/evidence/precision_dev/source'
        for p in SOURCES:
            destination=archive/p;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes((ROOT/p).read_bytes())
    print(json.dumps({k:metrics[k] for k in ['region_success_rate','over_redaction_rate','missed_regions','target_pass']}),flush=True)


if __name__=='__main__':main()
