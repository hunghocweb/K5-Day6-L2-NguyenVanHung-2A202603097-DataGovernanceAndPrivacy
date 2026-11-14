"""Routing isolation and a single explicit user padding operation."""
import json
import zipfile
import cv2
import numpy as np
import pytest
from src.t07_auto import AutoDetectors
from src.t07_detectors import clip_box


def box(**extra):
    return dict(x=20,y=20,w=20,h=20,type='face',confidence=.9,source='test',**extra)


@pytest.mark.parametrize('route',['real','2d','both'])
def test_dispatch_only_calls_selected_branch_without_padding(route):
    calls=[]
    class Photo:
        def detect_trace(self,image,progress):
            calls.append('real');return dict(raw=[box()],merged=[box()],views=[])
    def drawing(image):
        calls.append('2d');return dict(raw=[box()],merged=[box()])
    trace=AutoDetectors(photo_detector=Photo(),drawing_detector=drawing).detect_details(
        np.zeros((80,80,3),np.uint8),routing={'route':route})
    assert calls==(['real','2d'] if route=='both' else [route])
    assert [trace['merged'][0][k] for k in ['x','y','w','h']]==[20,20,20,20]
    assert trace['bbox_calibration_applied'] is False
    assert trace['detector_padding_applied'] is False


def test_native_faces_remove_legacy_padding_without_learned_calibration(monkeypatch):
    from src import t07_native_2d as native
    import src.t07_head_calibration as calibration
    def forbidden(*args,**kwargs):raise AssertionError('learned bbox correction must not run')
    for name in dir(calibration):
        if not name.startswith('_') and callable(getattr(calibration,name)):
            monkeypatch.setattr(calibration,name,forbidden)
    monkeypatch.setattr(native,'coarse_faces',lambda img:([dict(box(),x=19,y=19,w=22,h=22)],0))
    monkeypatch.setattr(native,'shape_faces',lambda img:([],0))
    monkeypatch.setattr(native,'tiny_faces',lambda img:([],0))
    monkeypatch.setattr(native,'shape_plates',lambda *a,**kw:[])
    monkeypatch.setattr(native,'verify_face_crop',lambda *a:True)
    monkeypatch.setattr(native,'restore_for_analysis',lambda img,**kw:(img,{'applied_sigma':0}))
    result=native.detect_details(np.zeros((80,80,3),np.uint8))
    assert [result['merged'][0][k] for k in ['x','y','w','h']]==[20,20,20,20]


@pytest.mark.parametrize('padding',[0,.10])
def test_auto_worker_padding_and_export_match_exactly_once(tmp_path,monkeypatch,padding):
    from tools import t07_auto_walk as worker
    from tools.t07_walkthrough import annotate
    from src.t07_pipeline import blur
    class Detector:
        def __init__(self,*a):pass
        def detect_details(self,*a,**kw):
            return dict(raw=[box()],merged=[box()],views=[],algorithm='test',review_reasons=[])
    monkeypatch.setattr(worker,'AutoDetectors',Detector)
    monkeypatch.setattr(worker,'route_image',lambda image:{'route':'2d'})
    image=np.random.default_rng(42).integers(0,255,(80,80,3),dtype=np.uint8)
    cv2.imwrite(str(tmp_path/'source.png'),image)
    worker.run_auto(tmp_path,dict(started=0,padding=padding,original_size=[80,80]),
        lambda *a,**kw:None,lambda value:None,annotate,blur)
    report=json.loads((tmp_path/'report.json').read_text(encoding='utf-8'))
    assert report['boxes']==[clip_box(box(),80,80,padding)]
    assert report['padding_applications']==1 and report['bbox_calibration_applied'] is False
    assert report['human_review'] is False
    assert np.array_equal(cv2.imread(str(tmp_path/'result.png')),blur(image,report['boxes']))
    with zipfile.ZipFile(tmp_path/'result.zip') as archive:
        assert set(archive.namelist())=={'result.png','redactions.csv','report.json'}
