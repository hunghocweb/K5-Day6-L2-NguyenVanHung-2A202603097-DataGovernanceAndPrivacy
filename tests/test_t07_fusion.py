from src.t07_fusion import fuse


def test_supported_text_tail_is_retained_without_copying_entire_coarse_car_box():
    coarse=[dict(x=10,y=10,w=100,h=30,type='plate',confidence=.9,source='coarse')]
    fine=[dict(x=30,y=18,w=40,h=12,type='plate',confidence=.8,source='panel')]
    boxes=fuse(coarse,fine,[],150,80,plate_x_fraction=.2)
    assert len(boxes)==1
    box=boxes[0]
    assert box['x']<=75<box['x']+box['w']  # supported text extends beyond substrate
    assert not box['x']<=12<box['x']+box['w']  # lamp at coarse box edge remains visible
    assert box['y']==18 and box['h']==12
