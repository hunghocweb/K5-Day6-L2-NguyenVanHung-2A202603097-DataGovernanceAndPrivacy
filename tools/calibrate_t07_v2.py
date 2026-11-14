"""Fit global geometry rules on accepted dev regions; no eval pixels or ID lookup."""
import itertools
import json
import zipfile
import io
import csv
from pathlib import Path

import cv2
import pandas as pd
import numpy as np

from src.t07_rules_v2 import detect
from src.t07_detectors import clip_box
from tools.compare_t07_review import mask


def main():
    images=pd.read_csv('fixtures/t07_redaction/images.csv')
    dev=set(images.loc[images.split=='dev','image_id'])
    with zipfile.ZipFile('out/web_review/t07_review_export.zip') as z:
        refs={r['item_id']:json.loads(r['override_value']) for r in csv.DictReader(io.StringIO(
            z.read('review_queue_reviewed.csv').decode('utf-8-sig')))
              if r['item_id'] in dev and r['reviewer_decision'] in ('approve','edit')}
    raw=json.loads(Path('out/poc/predictions_machine.json').read_text())
    candidates={}
    for key in refs:
        image=cv2.imread('fixtures/t07_redaction/images/'+key+'.jpg')
        candidates[key]=detect(image,[b for b in raw[key] if b['source'] in ('yunet','yolov9')])[0]
    results=[]
    for face_pad,plate_pad,drop_pairs in itertools.product((-1,0,1,2),(-2,-1,0,1),(False,True)):
        missed=n=extra=total=0
        for key,reference in refs.items():
            boxes=[]
            for b in candidates[key]:
                if drop_pairs and b['source'] in ('v2_paired_eye_components','v2_small_face_upscale'):continue
                pad=face_pad if b['type']=='face' else plate_pad
                if b['w']+2*pad<1 or b['h']+2*pad<1:continue
                boxes.append(clip_box({**b,'x':b['x']-pad,'y':b['y']-pad,'w':b['w']+2*pad,'h':b['h']+2*pad},640,360))
            pred=mask(boxes);gold=mask(reference)
            extra+=int((pred&~gold).sum());total+=int(pred.sum())
            for box in reference:
                region=mask([box]);n+=1
                missed+=int((pred&region).sum()/max(1,region.sum())<.9)
        results.append(dict(face_pad=face_pad,plate_pad=plate_pad,drop_pairs=drop_pairs,
                            missed=missed,n=n,extra_ratio=extra/max(1,total)))
    feasible=[r for r in results if r['extra_ratio']<=.3]
    chosen=min(feasible or results,key=lambda r:(r['missed'],r['extra_ratio']))
    output={'reference':'accepted dev review only, training agreement not independent quality',
            'dev_ids':sorted(refs),'selected':chosen,'candidates':results}
    Path('out/t07_v2_dev/calibration.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(chosen,indent=2))


if __name__=='__main__':main()
