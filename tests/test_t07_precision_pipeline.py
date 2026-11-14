"""Whole-cascade geometry checks on new drawings, not reviewed fixture crops."""
import cv2
import numpy as np
import pytest
from src.t07_precision_pipeline import detect_details
from src.metrics.detection import region_coverage


@pytest.mark.parametrize('sigma,scale,jpeg',[(0,1.,False),(1.5,1.2,True),(2.5,.8,True)])
def test_precision_cascade_keeps_pii_and_does_not_cover_empty_head(sigma,scale,jpeg):
    image=np.full((230,360,3),(70,120,75),np.uint8)
    for x in [60,285]:cv2.ellipse(image,(x,60),(18,24),0,0,360,(90,160,220),-1)
    for x in [54,66]:cv2.circle(image,(x,55),2,(20,20,20),-1)
    cv2.line(image,(55,70),(65,70),(20,20,20),2)
    cv2.rectangle(image,(90,145),(265,208),(180,85,55),-1)
    cv2.rectangle(image,(120,112),(235,148),(220,210,190),-1)
    cv2.rectangle(image,(145,174),(210,190),(240,240,240),-1)
    cv2.putText(image,'AB12',(150,187),cv2.FONT_HERSHEY_SIMPLEX,.4,(10,10,10),1,cv2.LINE_AA)
    if sigma:image=cv2.GaussianBlur(image,(0,0),sigma)
    image=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    image=cv2.copyMakeBorder(image,11,7,13,5,cv2.BORDER_CONSTANT,value=(70,120,75))
    if jpeg:
        ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,82]);assert ok
        image=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
    untouched=image.copy();result=detect_details(image);boxes=result['boxes']
    for (x,y,w,h),kind in [((42,36,37,49),'face'),((145,174,66,17),'plate')]:
        target=(x*scale+13,y*scale+11,w*scale,h*scale)
        candidates=[[b[k] for k in ['x','y','w','h']] for b in boxes if b['type']==kind]
        assert region_coverage(target,candidates,image.shape[:2])>=.90
    assert not any(b['type']=='face' and b['x']<=285*scale+13<b['x']+b['w']
        and b['y']<=60*scale+11<b['y']+b['h'] for b in boxes)
    assert np.array_equal(image,untouched)
    assert result['requires_review'] is True
