"""Recompute mathematical detections from pixels; guard one validation run."""
import argparse,json,hashlib,platform,time
from datetime import datetime,timezone
import cv2,numpy as np,pandas as pd
from src.t07_math2d import detect_details,Math2DConfig
from dataclasses import asdict
from tools.evaluate_t07_reviewed import ROOT,FIXTURE,EVIDENCE,read_rows,score,sha,write_json

DEPENDENCIES=['src/t07_math2d.py','src/t07_geometry.py','src/t07_local_shapes.py',
    'src/t07_fusion.py','src/t07_restoration.py','src/t07_rules_v3.py','src/t07_rules_v4.py',
    'src/t07_router.py','src/t07_detectors.py','src/metrics/detection.py',
    'tools/evaluate_t07_reviewed.py','tools/evaluate_t07_math2d.py']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['dev','eval'],default='dev')
    parser.add_argument('--reviewed-only',action='store_true');parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    if args.reviewed_only and (args.split!='dev' or args.freeze):raise ValueError('Review-only is for dev diagnostics')
    initial={p:sha(ROOT/p) for p in DEPENDENCIES};config=asdict(Math2DConfig())
    environment=dict(python=platform.python_version(),opencv=cv2.__version__,numpy=np.__version__,pandas=pd.__version__)
    frozen_path=ROOT/'reports/t07_math2d_freeze.json';once=ROOT/'reports/t07_math2d_validation_once.json'
    label_path=EVIDENCE/f'regions_{args.split}.csv'
    if args.split=='eval':
        frozen=json.loads(frozen_path.read_text(encoding='utf-8'))
        if frozen['source_sha256']!=initial or frozen['config']!=json.loads(json.dumps(config)) or frozen['environment']!=environment:
            raise ValueError('Code, configuration or environment differs from dev freeze')
        if not frozen['dev_metrics']['guardrail_pass'] or frozen['dev_metrics']['region_success_rate']<.95:
            raise ValueError('Dev target has not been met')
        if frozen['dev_reference_sha256']!=sha(EVIDENCE/'regions_dev.csv'):raise ValueError('Dev labels changed')
        if once.exists():raise ValueError('Validation already started; no retuning or retry')
        # Reserve before loading any eval pixels/labels, even if the run fails.
        with once.open('x',encoding='utf-8') as handle:json.dump(dict(state='started',at=datetime.now(timezone.utc).isoformat()),handle)
        if not label_path.exists():
            pd.DataFrame([r for r in read_rows(EVIDENCE/'regions.csv') if r['split']=='eval']).to_csv(label_path,index=False)
    reference=pd.read_csv(label_path)
    images={r['image_id']:r for r in read_rows(FIXTURE/'images.csv') if r['split']==args.split}
    if args.reviewed_only:images={k:r for k,r in images.items() if k in set(reference.image_id)}
    keys=list(reference.image_id.unique());sizes={k:(int(r['height']),int(r['width'])) for k,r in images.items()}
    out=ROOT/f'out/t07_math2d_{args.split}';out.mkdir(exist_ok=True)
    start=time.perf_counter();predictions={};trace={}
    for index,(key,row) in enumerate(images.items()):
        image=cv2.imread(str(FIXTURE/row['path']))
        if image is None or image.shape[:2]!=sizes[key]:raise ValueError(f'Invalid source image {key}')
        result=detect_details(image);predictions[key]=result['boxes']
        trace[key]={k:v for k,v in result.items() if k not in ['boxes','stages']}
        trace[key]['image_sha256']=sha(FIXTURE/row['path'])
        write_json(out/'progress.json',dict(processed=index+1,total=len(images),image_id=key))
        print(f'{args.split} {index+1}/{len(images)} {key}',flush=True)
    if {p:sha(ROOT/p) for p in DEPENDENCIES}!=initial:raise RuntimeError('Source changed during inference')
    metrics,coverage=score(reference,predictions,keys,sizes)
    metrics.update(split=args.split,seconds=time.perf_counter()-start,n_processed_images=len(images),seed=42,
        source_sha256=initial,config=config,environment=environment,reference_sha256=sha(label_path),independent_holdout=False)
    write_json(out/'predictions.json',predictions);write_json(out/'trace.json',trace)
    with (out/'machine_audit.jsonl').open('w',encoding='utf-8') as handle:
        for key in images:
            handle.write(json.dumps(dict(event='machine_detection',actor_kind='machine',image_id=key,
                route='2d',algorithm=trace[key]['algorithm'],image_sha256=trace[key]['image_sha256'],
                source_sha256=initial,requires_human_review=True,human_review_recorded=False,
                boxes=predictions[key]),ensure_ascii=False)+'\n')
    write_json(out/'metrics.json',metrics);coverage.to_csv(out/'coverage.csv',index=False)
    pd.DataFrame([dict(image_id=k,**b) for k,boxes in predictions.items() for b in boxes]).to_csv(out/'predictions.csv',index=False)
    if args.freeze:
        if args.split!='dev' or args.reviewed_only:raise ValueError('Freeze must process the full dev split')
        if metrics['region_success_rate']<.95 or not metrics['guardrail_pass']:raise ValueError('Dev targets not met; validation remains closed')
        write_json(frozen_path,dict(at=datetime.now(timezone.utc).isoformat(),source_sha256=initial,config=config,environment=environment,
            dev_reference_sha256=sha(label_path),dev_metrics=metrics,warning='Prior data exposure: validation is not a pristine independent holdout'))
        archive=ROOT/'reports/evidence/math2d';archive.mkdir(exist_ok=True)
        for path in DEPENDENCIES:
            destination=archive/'source'/path;destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes((ROOT/path).read_bytes())
    if args.split=='eval':write_json(once,dict(state='complete',metrics=metrics))
    print(json.dumps({k:metrics[k] for k in ['region_success_rate','over_redaction_rate','missed_regions','guardrail_pass','by_type']}),flush=True)


if __name__=='__main__':main()
