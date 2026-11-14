import cv2
import numpy as np
import pytest
from src.t07_face_evidence import joint_facial_score
from src.t07_text_extent import paired_window_above_body,profile_panel_bounds


@pytest.mark.parametrize('blur',[0.,1.5,2.5])
def test_window_is_above_car_body_and_is_not_itself_a_car_body(blur):
    image=np.full((220,320,3),(70,120,75),np.uint8)
    cv2.rectangle(image,(60,100),(240,165),(180,85,55),-1)
    cv2.rectangle(image,(90,65),(210,105),(220,210,190),-1)
    if blur:image=cv2.GaussianBlur(image,(0,0),blur)
    assert paired_window_above_body(image,[60,100,181,66])
    assert not paired_window_above_body(image,[90,65,121,41])


@pytest.mark.parametrize('spacing',[2,3])
def test_tiny_joint_marks_have_more_evidence_than_empty_head(spacing):
    image=np.full((14,12),90,np.uint8)
    cv2.ellipse(image,(6,7),(4,5),0,0,360,190,-1)
    blank=image.copy()
    for x in [6-spacing,6+spacing]:cv2.circle(image,(x,5),0,20,-1)
    cv2.line(image,(4,9),(8,9),20,1)
    assert joint_facial_score(image)>joint_facial_score(blank)+.3


def test_glyph_profile_does_not_follow_constant_dark_background():
    image=np.full((100,240,3),(110,100,160),np.uint8)
    cv2.rectangle(image,(80,40),(130,55),(240,240,240),-1)
    cv2.putText(image,'AB12345',(84,51),cv2.FONT_HERSHEY_SIMPLEX,.35,(20,20,20),1,cv2.LINE_AA)
    image[:,180:]=(10,10,10)
    seed=dict(x=72,y=36,w=67,h=24,type='plate',confidence=.8,panel_xywh=[80,40,51,16])
    result=profile_panel_bounds(image,seed,threshold=1,margin_x=1,margin_y=1)
    assert result['x']+result['w']<180
