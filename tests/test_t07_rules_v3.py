"""Independent toy geometry checks; no instructor GT or fixture coordinates."""
import cv2
import numpy as np

from src.t07_rules_v3 import faces,plates,extend_plate_text


def test_ellipse_needs_facial_marks_and_does_not_redact_blank_back_head():
    image=np.full((180,260,3),(70,80,90),np.uint8)
    for x in [65,190]:cv2.ellipse(image,(x,85),(22,29),0,0,360,(80,150,210),-1)
    for x in [57,73]:cv2.circle(image,(x,80),2,(15,15,15),-1)
    cv2.line(image,(59,96),(71,96),(15,15,15),2)
    boxes,_=faces(image)
    assert any(b['x']<=65<=b['x']+b['w'] and b['y']<=85<=b['y']+b['h'] for b in boxes)
    assert not any(b['x']<=190<=b['x']+b['w'] for b in boxes)


def test_plate_text_extension_includes_black_glyphs_on_both_sides():
    image=np.full((80,220,3),(190,130,180),np.uint8)
    cv2.rectangle(image,(75,30),(130,50),(245,245,245),-1)
    cv2.putText(image,'A12B34C',(53,46),cv2.FONT_HERSHEY_SIMPLEX,.55,(10,10,10),1,cv2.LINE_AA)
    box=extend_plate_text(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(75,30,56,21),image)
    assert box[0]<75 and box[0]+box[2]>131


def test_empty_white_rectangle_is_not_a_plate():
    image=np.full((100,200,3),80,np.uint8)
    cv2.rectangle(image,(50,40),(120,60),(245,245,245),-1)
    assert plates(image)==[]


def test_single_stroke_digit_outside_substrate_is_kept():
    image=np.full((80,220,3),180,np.uint8)
    cv2.rectangle(image,(75,30),(130,50),(245,245,245),-1)
    cv2.line(image,(136,35),(136,45),(10,10,10),1)
    box=extend_plate_text(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(75,30,56,21),image)
    assert box[0]+box[2]>136


def test_glyph_height_may_exceed_the_white_substrate_height():
    image=np.full((80,220,3),170,np.uint8)
    cv2.rectangle(image,(75,35),(99,40),(245,245,245),-1)
    cv2.putText(image,'12345',(78,43),cv2.FONT_HERSHEY_SIMPLEX,.35,(5,5,5),1,cv2.LINE_AA)
    box=extend_plate_text(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(75,35,25,6),image)
    assert box[0]+box[2]>100 and box[1]+box[3]>41
