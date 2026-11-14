import numpy as np
from src.t07_multiscale import MultiScaleDetectors, windows


def test_windows_cover_every_pixel_and_overlap():
    coverage=np.zeros((721,1281),dtype=np.uint8)
    for x,y,w,h in windows(1281,721):coverage[y:y+h,x:x+w]+=1
    assert coverage.min()>=1
    assert coverage.max()>1


def test_letterbox_preserves_geometry_and_ignores_padding_predictions():
    class Face:
        def detect(self,image):
            assert image.shape==(320,320,3)
            assert np.all(image[:70]==114)
            return None,np.array([[50,95,20,30,.95],[10,5,20,20,.99]])
    class Plate:
        def predict(self,image):return []
    detector=MultiScaleDetectors.__new__(MultiScaleDetectors)
    detector.face,detector.plate=Face(),Plate()
    boxes=detector.detect_view(np.zeros((360,640,3),dtype=np.uint8))
    assert len(boxes)==1
    assert [boxes[0][k] for k in ('x','y','w','h')]==[100,50,40,60]


def test_crop_boxes_map_back_and_deduplicate():
    detector=MultiScaleDetectors.__new__(MultiScaleDetectors)
    detector.detect_view=lambda image:[dict(x=100,y=100,w=20,h=30,type='face',confidence=.9,source='test')]
    boxes=detector.detect(np.zeros((360,640,3),dtype=np.uint8))
    assert any(b['x']==420 and b['y']==140 for b in boxes)
    assert len({(b['x'],b['y'],b['w'],b['h']) for b in boxes})==len(boxes)


def test_tile_edge_only_detection_is_kept_for_review():
    detector=MultiScaleDetectors.__new__(MultiScaleDetectors)
    calls=[]
    def view(image):
        calls.append(1)
        return [dict(x=300,y=50,w=20,h=20,type='plate',confidence=.9)] if len(calls)==2 else []
    detector.detect_view=view
    trace=detector.detect_trace(np.zeros((320,640,3),np.uint8))
    assert len(trace['merged'])==1 and trace['merged'][0]['x']==300
    assert trace['partial_fallbacks'][0]['requires_review'] is True


def test_complete_box_wins_over_high_confidence_edge_fragment():
    detector=MultiScaleDetectors.__new__(MultiScaleDetectors)
    calls=[]
    def view(image):
        calls.append(1)
        if len(calls)==1:return [dict(x=300,y=50,w=40,h=20,type='plate',confidence=.7)]
        if len(calls)==2:return [dict(x=300,y=50,w=20,h=20,type='plate',confidence=.99)]
        return []
    detector.detect_view=view
    trace=detector.detect_trace(np.zeros((320,640,3),np.uint8))
    assert len(trace['raw'])==2 and len(trace['merged'])==1
    assert trace['merged'][0]['w']==40 and trace['partial_fallbacks']==[]


def test_other_class_does_not_suppress_edge_detection():
    detector=MultiScaleDetectors.__new__(MultiScaleDetectors)
    calls=[]
    def view(image):
        calls.append(1)
        if len(calls)==1:return [dict(x=300,y=50,w=40,h=20,type='face',confidence=.9)]
        if len(calls)==2:return [dict(x=300,y=50,w=20,h=20,type='plate',confidence=.8)]
        return []
    detector.detect_view=view
    trace=detector.detect_trace(np.zeros((320,640,3),np.uint8))
    assert {b['type'] for b in trace['merged']}=={'face','plate'}
