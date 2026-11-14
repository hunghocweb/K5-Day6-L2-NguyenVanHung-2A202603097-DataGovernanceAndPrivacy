"""Reproduce frozen 2D primary and its pixel gate on dev only."""
import argparse,csv,hashlib,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import cv2
import pandas as pd
from src.t07_cascade_2d import primary_suitability
from src.t07_detectors import clip_box
from tools.evaluate_t07_reviewed import score,write_json

ROOT=Path(__file__).resolve().parents[1]
SOURCES=['src/t07_cascade_2d.py','src/t07_auto.py','src/t07_math2d.py','src/t07_fusion.py',
         'src/t07_restoration.py','src/t07_geometry.py','src/t07_local_shapes.py','src/t07_rules_v3.py',
         'src/t07_rules_v4.py','src/t07_detectors.py','src/t07_router.py','src/t07_native_2d.py',
         'src/t07_drawing_details.py','src/t07_face_evidence.py','src/t07_precision.py',
         'src/t07_text_extent.py','tools/verify_t07_cascade_dev.py']


def hashes():return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}


def infer(row):
    cv2.setNumThreads(1)
    from src.t07_auto import AutoDetectors
    image=cv2.imread(str(ROOT/'fixtures/t07_redaction'/row['path']))
    if image is None:raise ValueError('Unreadable dev image')
    return row['image_id'],AutoDetectors().detect_details(image)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reviewed-only',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'out/t07_cascade_dev')
    args=parser.parse_args();output=args.output.resolve()
    if not output.is_relative_to((ROOT/'out').resolve()):parser.error('Output must be under out')
    output.mkdir(parents=True,exist_ok=True);before=hashes();begun=time.perf_counter()
    with (ROOT/'fixtures/t07_redaction/images.csv').open(encoding='utf-8-sig') as stream:
        dev=[r for r in csv.DictReader(stream) if r['split']=='dev']
    cv2.setNumThreads(1)
    gates={r['image_id']:primary_suitability(cv2.imread(str(ROOT/'fixtures/t07_redaction'/r['path']))) for r in dev}
    write_json(output/'gate_all_dev.json',gates)
    print('gate dev',dict(primary_suitable=sum(g['suitable'] for g in gates.values()),total=len(gates)),flush=True)
    reference=pd.read_csv(ROOT/'reports/evidence/reviewed_reference/regions_dev.csv')
    keys=list(reference.image_id.unique());rows=[r for r in dev if not args.reviewed_only or r['image_id'] in keys]
    results={}
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(infer,r) for r in rows]
        for future in as_completed(futures):
            key,result=future.result();results[key]=result
            write_json(output/'progress.json',dict(processed=len(results),total=len(rows),image_id=key))
            print(f'dev {len(results)}/{len(rows)} {key} {result["selected_algorithm"]}',flush=True)
    if hashes()!=before:raise RuntimeError('Sources changed during inference')
    write_json(output/'detections.json',results);metrics={}
    sizes={r['image_id']:(int(r['height']),int(r['width'])) for r in rows}
    for padding in [0.,.06]:
        predictions={k:[clip_box(b,sizes[k][1],sizes[k][0],padding) for b in r['merged']] for k,r in results.items()}
        metric,coverage=score(reference,predictions,keys,sizes)
        metrics[str(padding)]=metric;coverage.to_csv(output/f'coverage_pad{padding}.csv',index=False)
        print(padding,{k:metric[k] for k in ['region_success_rate','over_redaction_rate','n_predictions']},flush=True)
    write_json(output/'metrics.json',dict(metrics=metrics,source_sha256=before,n_processed=len(rows),
        n_labelled_images=len(keys),seconds=time.perf_counter()-begun,independent_test=False,
        additional_padding_default=0.,original_reference='user_reviewed_dev_boxes',
        gate_is_accuracy=False,gate_thresholds=dict(next(iter(gates.values()))['config'])))


if __name__=='__main__':main()
