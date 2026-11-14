"""Dev-only image rule experiment, no model or eval pixels."""
import json
from pathlib import Path

import cv2
import pandas as pd

from src.t07_rules_v2 import detect
from src.t07_detectors import clip_box
from src.t07_pipeline import blur
from tools.compare_t07_review import compare


def main():
    output=Path("out/t07_v2_dev")
    (output/"redacted").mkdir(parents=True,exist_ok=True)
    rows=[]
    raw=json.loads(Path("out/poc/predictions_machine.json").read_text(encoding="utf-8"))
    for row in pd.read_csv("fixtures/t07_redaction/images.csv").itertuples():
        if row.split!="dev":continue
        image=cv2.imread(str(Path("fixtures/t07_redaction")/row.path))
        pretrained=[b for b in raw[row.image_id] if b["source"] in {"yolov9","yunet"}]
        boxes,_=detect(image,pretrained)
        boxes=[clip_box(b,row.width,row.height,.03) for b in boxes]
        rows.extend({"image_id":row.image_id,**b} for b in boxes)
        cv2.imwrite(str(output/"redacted"/f"{row.image_id}.png"),blur(image,boxes))
    pd.DataFrame(rows).to_csv(output/"redactions.csv",index=False)
    results={name:compare(path,"out/web_review/t07_review_export.zip") for name,path in {
        "v1":"out/poc/redactions.csv","v2":output/"redactions.csv"}.items()}
    (output/"comparison.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(json.dumps({k:{name:v for name,v in result.items() if name!="per_image"} for k,result in results.items()},indent=2))


if __name__=="__main__":main()
