"""Render only labelled dev examples for visual inspection; no review actions."""
import json,cv2,numpy as np,pandas as pd
from tools.evaluate_t07_reviewed import ROOT,FIXTURE,EVIDENCE
from src.t07_pipeline import blur


def main():
    ref=pd.read_csv(EVIDENCE/'regions_dev.csv');keys=list(ref.image_id.unique())
    pred=json.loads((ROOT/'out/t07_precision_pixel_dev/predictions.json').read_text())
    coverage=pd.read_csv(ROOT/'out/t07_precision_pixel_dev/coverage.csv')
    out=ROOT/'out/t07_precision_visual_dev';out.mkdir(exist_ok=True)
    tiles=[]
    for key in keys:
        image=cv2.imread(str(FIXTURE/f'images/{key}.jpg'));result=blur(image,pred[key])
        cv2.imwrite(str(out/f'{key}_result.png'),result)
        tile=np.full((390,640,3),245,np.uint8);tile[30:]=result
        cv2.putText(tile,key,(12,22),cv2.FONT_HERSHEY_SIMPLEX,.65,(25,25,25),1,cv2.LINE_AA);tiles.append(tile)
        for gt in coverage[(coverage.image_id==key)&(coverage.missed==1)].itertuples():
            xa,ya=max(0,gt.x-8),max(0,gt.y-8);xb,yb=min(640,gt.x+gt.w+8),min(360,gt.y+gt.h+8)
            raw=image[ya:yb,xa:xb];redacted=result[ya:yb,xa:xb];annotated=redacted.copy()
            cv2.rectangle(annotated,(gt.x-xa,gt.y-ya),(gt.x+gt.w-xa-1,gt.y+gt.h-ya-1),(0,0,255),1)
            panels=np.concatenate([raw,redacted,annotated],axis=1)
            cv2.imwrite(str(out/f'{key}_{gt.type}_undercoverage.png'),cv2.resize(panels,None,fx=6,fy=6,interpolation=cv2.INTER_NEAREST))
    sheet=np.concatenate([np.concatenate(tiles[i:i+3],axis=1) for i in range(0,len(tiles),3)],axis=0)
    cv2.imwrite(str(out/'contact_sheet.png'),sheet)
    print(f'Rendered {len(keys)} labelled dev images, with {int(coverage.missed.sum())} undercoverage crops',flush=True)


if __name__=='__main__':main()
