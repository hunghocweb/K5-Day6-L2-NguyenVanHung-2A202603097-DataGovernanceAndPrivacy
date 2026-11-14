"""Generic cartoon geometry regressions; no fixture identities or GT lookup."""
import cv2
import numpy as np
import pytest

from src.t07_rules_v4 import faces, plates, extend_plate_text


def face_scene(blur=0):
    image=np.full((150,230,3),(70,80,90),np.uint8)
    for x in [60,175]:cv2.ellipse(image,(x,70),(18,24),0,0,360,(90,160,220),-1)
    for x in [54,66]:cv2.circle(image,(x,65),2,(30,30,30),-1)
    cv2.line(image,(55,80),(65,80),(30,30,30),2)
    return cv2.GaussianBlur(image,(0,0),blur) if blur else image


def test_face_marks_survive_blur_but_blank_back_head_is_not_redacted():
    for sigma in [0,1.5]:
        boxes,_=faces(face_scene(sigma))
        assert any(b['x']<=60<b['x']+b['w'] and b['y']<=70<b['y']+b['h'] for b in boxes)
        assert not any(b['x']<=175<b['x']+b['w'] for b in boxes)


def test_blank_white_rectangle_is_not_plate():
    image=np.full((100,220,3),100,np.uint8)
    cv2.rectangle(image,(60,40),(140,60),(245,245,245),-1)
    assert plates(image)==[]


def test_plate_growth_does_not_include_red_taillights():
    image=np.full((80,220,3),(90,80,170),np.uint8)
    cv2.rectangle(image,(70,30),(135,47),(240,240,240),-1)
    cv2.putText(image,'AB123',(72,43),cv2.FONT_HERSHEY_SIMPLEX,.4,(10,10,10),1)
    cv2.ellipse(image,(150,34),(7,4),0,0,360,(0,0,255),-1)
    x,y,w,h=extend_plate_text(image,(70,30,66,18))
    assert x+w<143
    assert h<36  # includes safety padding, excludes growth into an entire car body


@pytest.mark.parametrize('scale',[.65,1.,1.4])
@pytest.mark.parametrize('skin',[(50,95,145),(100,175,235)])
def test_face_translation_scale_skin_and_jpeg_change_without_identity_lookup(scale,skin):
    image=face_scene(1.0)
    # Recolour both heads and redraw the face so that colour is not a label.
    image[:]=[65,80,70]
    for x in [60,175]:cv2.ellipse(image,(x,70),(18,24),0,0,360,skin,-1)
    for x in [54,66]:cv2.circle(image,(x,65),2,(15,15,15),-1)
    cv2.line(image,(55,80),(65,80),(15,15,15),2)
    image=cv2.GaussianBlur(image,(0,0),1.)
    image=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    image=cv2.copyMakeBorder(image,9,3,17,5,cv2.BORDER_CONSTANT,value=(45,65,80))
    ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,82]);assert ok
    image=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
    boxes,_=faces(image)
    cx,cy=60*scale+17,70*scale+9
    assert any(b['x']<=cx<b['x']+b['w'] and b['y']<=cy<b['y']+b['h'] for b in boxes)
    back=175*scale+17
    assert not any(b['x']<=back<b['x']+b['w'] and b['y']<=cy<b['y']+b['h'] for b in boxes)


@pytest.mark.parametrize('brightness',[.45,1.])
def test_blurred_plate_with_arbitrary_text_and_empty_window(brightness):
    image=np.full((140,260,3),(90,100,180),np.uint8)
    cv2.rectangle(image,(70,20),(180,42),(210,210,210),-1)
    cv2.rectangle(image,(85,80),(160,97),(240,240,240),-1)
    cv2.putText(image,'AB12C',(88,93),cv2.FONT_HERSHEY_SIMPLEX,.43,(10,10,10),1,cv2.LINE_AA)
    image=cv2.GaussianBlur(image,(0,0),1.1)
    image=np.clip(image.astype(float)*brightness,0,255).astype(np.uint8)
    boxes=plates(image)
    assert any(b['x']<=120<b['x']+b['w'] and b['y']<=87<b['y']+b['h'] for b in boxes)
    assert not any(b['x']<=120<b['x']+b['w'] and b['y']<=30<b['y']+b['h'] for b in boxes)
