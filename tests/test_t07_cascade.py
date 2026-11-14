"""Fallback decisions depend on pixels/output evidence, never unknown accuracy."""
import cv2
import numpy as np
from src.t07_cascade_2d import Cascade2D,primary_suitability
from src.t07_auto import AutoDetectors


def box():return dict(x=15,y=15,w=12,h=14,type='face',confidence=.8,source='test')


def flat_image():
    image=np.full((100,160,3),(120,160,90),np.uint8)
    cv2.ellipse(image,(70,40),(10,14),0,0,360,(70,130,210),-1)
    return image


def test_supported_drawing_keeps_frozen_primary_regions_exactly():
    calls=[]
    def primary(image):calls.append('primary');return dict(boxes=[box()],algorithm='math2d-dev-1')
    def fallback(image):raise AssertionError('Supported nonempty primary must not be replaced')
    image=flat_image();before=image.copy()
    result=Cascade2D(primary,fallback)(image)
    assert result['merged']==[box()] and calls==['primary']
    assert result['detector_padding_applied'] is True
    assert result['cascade']['selected']=='ellipse_blur_2d'
    assert result['cascade']['gate']['measured_accuracy'] is None
    assert np.array_equal(image,before)


def test_empty_primary_runs_detailed_fallback_and_keeps_reason():
    calls=[]
    def primary(image):calls.append('primary');return dict(boxes=[])
    def fallback(image):calls.append('fallback');return dict(raw=[box()],merged=[box()],ambiguous_heads=1)
    result=Cascade2D(primary,fallback)(flat_image())
    assert calls==['primary','fallback'] and result['merged']==[box()]
    assert result['cascade']['reason']=='primary_empty_not_a_safety_verdict'
    assert result['detector_padding_applied'] is False


def test_detailed_pixels_skip_unsuitable_primary_and_do_not_load_photo_model():
    image=np.random.default_rng(42).integers(0,255,(100,160,3),dtype=np.uint8)
    def primary(image):raise AssertionError('Unsuitable primary should not execute')
    def fallback(image):return dict(raw=[box()],merged=[box()],views=[])
    class Photo:
        def detect_trace(self,*args):raise AssertionError('No photo model for a 2D route')
    result=AutoDetectors(photo_detector=Photo(),drawing_detector=Cascade2D(primary,fallback)).detect_details(
        image,routing={'route':'2d'})
    assert result['cascade']['primary_executed'] is False
    assert result['cascade']['selected']=='detailed_2d'
    assert result['merged']==[box()]


def test_black_frame_does_not_make_detail_gate_claim_flat_content():
    image=np.random.default_rng(42).integers(0,255,(100,160,3),dtype=np.uint8)
    framed=cv2.copyMakeBorder(image,25,25,25,25,cv2.BORDER_CONSTANT,value=(0,0,0))
    first,second=primary_suitability(image),primary_suitability(framed)
    assert first['suitable']==second['suitable']==False
    assert first['palette_mass']==second['palette_mass']
    assert first['edge_fraction']==second['edge_fraction']
