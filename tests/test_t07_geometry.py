"""Geometry tests generated independently of the reviewed fixture images."""
import cv2
import numpy as np
import pytest

from src.t07_geometry import faces,plates,complete_ellipse
from src.t07_rules_v4 import _head_likelihood


@pytest.mark.parametrize('blur',[0.,1.5,2.5])
def test_actual_ellipse_face_beats_same_empty_ellipse_after_blur(blur):
    image=np.full((150,230,3),(70,80,90),np.uint8)
    for x in [60,175]:cv2.ellipse(image,(x,70),(18,24),0,0,360,(90,160,220),-1)
    for x in [54,66]:cv2.circle(image,(x,65),2,(30,30,30),-1)
    cv2.line(image,(55,80),(65,80),(30,30,30),2)
    if blur:image=cv2.GaussianBlur(image,(0,0),blur)
    boxes,_=faces(image,_head_likelihood)
    assert any(b['x']<=60<b['x']+b['w'] and b['y']<=70<b['y']+b['h'] for b in boxes)
    assert not any(b['x']<=175<b['x']+b['w'] and b['y']<=70<b['y']+b['h'] for b in boxes)


@pytest.mark.parametrize('blur',[0.,1.5,2.5])
def test_bounded_car_panel_survives_blur(blur):
    image=np.full((220,320,3),(70,120,75),np.uint8)
    cv2.rectangle(image,(60,100),(240,165),(180,85,55),-1)
    cv2.rectangle(image,(90,65),(210,105),(220,210,190),-1)
    cv2.rectangle(image,(120,128),(180,144),(240,240,240),-1)
    cv2.putText(image,'AB12',(123,140),cv2.FONT_HERSHEY_SIMPLEX,.40,(10,10,10),1,cv2.LINE_AA)
    if blur:image=cv2.GaussianBlur(image,(0,0),blur)
    boxes=plates(image)
    assert any(b['x']<=150<b['x']+b['w'] and b['y']<=136<b['y']+b['h'] for b in boxes)


def test_ransac_completes_curved_arc_without_fitting_occluder_line():
    field=np.zeros((140,140),np.uint8)
    cv2.ellipse(field,(70,70),(20,28),0,0,360,1,-1)
    field[:,73:]=0
    contours,_=cv2.findContours(field,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    fit=complete_ellipse(contours[0]);assert fit is not None
    (x,y),(a,b),_=fit
    assert abs(x-70)<4 and abs(y-70)<4
    assert 35<min(a,b)<47 and 48<max(a,b)<64
