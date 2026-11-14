import cv2
import numpy as np
from src.t07_drawing_details import plates,faces


def scene():
    return np.full((300,400,3),(55,80,65),np.uint8)


def test_text_panel_without_coloured_car_body_is_detected():
    image=scene()
    cv2.rectangle(image,(100,90),(190,115),(240,240,240),-1)
    cv2.putText(image,'AB-123',(105,108),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1,cv2.LINE_AA)
    found=plates(image)
    assert any(b['x']<=110 and b['x']+b['w']>=180 for b in found)


def test_empty_panels_lamps_and_coloured_icons_are_not_plates():
    image=scene()
    cv2.rectangle(image,(20,30),(100,50),(240,240,240),-1)
    cv2.circle(image,(170,70),10,(240,240,240),-1)
    cv2.rectangle(image,(210,95),(280,120),(245,245,245),-1)
    cv2.putText(image,'BUS',(212,114),cv2.FONT_HERSHEY_SIMPLEX,.5,(200,60,20),2)
    assert plates(image)==[]


def draw_head(image,cx,cy,marks=True):
    cv2.ellipse(image,(cx,cy),(24,31),0,0,360,(30,25,25),-1)
    cv2.ellipse(image,(cx,cy+6),(18,23),0,0,360,(176,190,236),-1)
    if marks:
        for dx in [-7,7]:cv2.circle(image,(cx+dx,cy+5),2,(20,20,20),-1)
        cv2.ellipse(image,(cx,cy+17),(3,1),0,0,360,(35,35,45),-1)


def test_faces_follow_translation_and_reject_blank_back_of_head():
    for cx,cy in [(90,75),(210,120)]:
        image=scene();draw_head(image,cx,cy)
        detected=faces(image)
        assert any(b['x']<cx<b['x']+b['w'] and b['y']<cy+5<b['y']+b['h'] for b in detected)
    image=scene();draw_head(image,90,75,marks=False)
    assert faces(image)==[]
