"""Independent geometry, not fixture identities or reviewed coordinates."""
import cv2
import numpy as np
import pytest
from src.t07_precision import PrecisionConfig,refit_face,refit_plate,remove_fragments


@pytest.mark.parametrize('sigma',[0.,1.5,2.5])
def test_face_refit_follows_colour_boundary_instead_of_large_proposal(sigma):
    image=np.full((140,180,3),(70,120,75),np.uint8)
    cv2.ellipse(image,(75,65),(18,24),0,0,360,(90,160,220),-1)
    for x in [69,81]:cv2.circle(image,(x,60),2,(20,20,20),-1)
    cv2.line(image,(70,75),(80,75),(20,20,20),2)
    if sigma:image=cv2.GaussianBlur(image,(0,0),sigma)
    box=dict(x=49,y=32,w=53,h=66,type='face',confidence=.8,source='test_proposal')
    result=refit_face(image,box,PrecisionConfig())
    assert result['refinement']=='face_colour_half_level'
    assert abs(result['x']-56)<=2 and abs(result['y']-40)<=2
    assert result['w']*result['h']<box['w']*box['h']*.75
    assert result['x']+result['w']>=92 and result['y']+result['h']>=88


@pytest.mark.parametrize('sigma',[0.,1.5,2.5])
def test_plate_refit_removes_margin_without_losing_substrate(sigma):
    image=np.full((220,320,3),(70,120,75),np.uint8)
    cv2.rectangle(image,(60,100),(240,165),(180,85,55),-1)
    cv2.rectangle(image,(90,65),(210,105),(220,210,190),-1)
    cv2.rectangle(image,(120,128),(180,144),(240,240,240),-1)
    cv2.putText(image,'AB12',(123,140),cv2.FONT_HERSHEY_SIMPLEX,.40,(10,10,10),1,cv2.LINE_AA)
    if sigma:image=cv2.GaussianBlur(image,(0,0),sigma)
    box=dict(x=112,y=124,w=77,h=25,type='plate',confidence=.8,source='test_proposal',
             body_xywh=[60,100,181,66],panel_xywh=[120,128,61,17])
    result=refit_plate(image,box,PrecisionConfig())
    assert result['refinement']=='plate_neutral_half_level'
    assert abs(result['x']-118)<=2 and abs(result['y']-127)<=2
    assert result['w']*result['h']<box['w']*box['h']*.8
    assert result['x']+result['w']>=180 and result['y']+result['h']>=144


def test_enclosed_feature_does_not_change_head_or_neighbour():
    def box(x,y,w,h):return dict(x=x,y=y,w=w,h=h,type='face')
    head=box(10,10,20,30);feature=box(14,16,4,4);neighbour=box(25,12,20,30)
    assert remove_fragments([head,feature,neighbour])==[head,neighbour]
