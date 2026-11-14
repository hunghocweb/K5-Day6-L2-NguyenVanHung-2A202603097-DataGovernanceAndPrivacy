"""Compare dev predictions to accepted user review, not instructor GT."""
import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np


def mask(boxes, width=640, height=360):
    result = np.zeros((height,width),dtype=bool)
    for b in boxes:
        x,y,w,h = (int(float(b[k])) for k in ("x","y","w","h"))
        result[y:y+h,x:x+w] = True
    return result


def compare(pred_path, review_zip, images_path="fixtures/t07_redaction/images.csv"):
    with Path(images_path).open(encoding="utf-8-sig") as f:
        dev = {row["image_id"] for row in csv.DictReader(f) if row["split"] == "dev"}
    with zipfile.ZipFile(review_zip) as archive:
        refs = {r["item_id"]: json.loads(r["override_value"]) for r in csv.DictReader(
            io.StringIO(archive.read("review_queue_reviewed.csv").decode("utf-8-sig")))
                if r["item_id"] in dev and r["reviewer_decision"] in {"approve","edit"}}
    predictions = {}
    with Path(pred_path).open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            predictions.setdefault(row["image_id"],[]).append(row)
    details=[]
    for key, boxes in refs.items():
        detected=predictions.get(key,[])
        union=mask(detected)
        missed=0
        coverage=[]
        for b in boxes:
            region=mask([b]); c=float((union & region).sum()/max(1,region.sum()))
            coverage.append(round(c,4));missed+=c<.9
        ref=mask(boxes)
        details.append({"image_id":key,"reference_regions":len(boxes),"missed_reference_regions":missed,
                        "coverage":coverage,"extra_pixels":int((union & ~ref).sum()),
                        "redacted_pixels":int(union.sum())})
    n=sum(r["reference_regions"] for r in details)
    return {"reference":"User-accepted dev review; previously corrected reference, not independent instructor GT",
            "split":"dev","n_images":len(details),"n_reference_regions":n,
            "missed_reference_regions":sum(r["missed_reference_regions"] for r in details),
            "reference_miss_rate":sum(r["missed_reference_regions"] for r in details)/max(n,1),
            "extra_pixel_ratio_vs_reference":sum(r["extra_pixels"] for r in details)/max(1,sum(r["redacted_pixels"] for r in details)),
            "per_image":details,"official_quality_metrics":None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred",required=True)
    p.add_argument("--review-zip",default="out/web_review/t07_review_export.zip")
    p.add_argument("--out",required=True)
    a=p.parse_args()
    result=compare(a.pred,a.review_zip)
    Path(a.out).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="per_image"},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
