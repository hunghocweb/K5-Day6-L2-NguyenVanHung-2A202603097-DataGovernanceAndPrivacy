"""Experimental pixel-only precision cascade; no image IDs or label access."""
import json
from functools import lru_cache
from pathlib import Path
from .t07_math2d import detect_details as base_detect
from .t07_precision_heads import refine_heads
from .t07_head_calibration import predict as calibrate_head
from .t07_text_extent import profile_panel_bounds
from .t07_restoration import estimate_sigma
from .t07_detectors import clip_box


@lru_cache(maxsize=1)
def load_bundle():
    return json.loads((Path(__file__).resolve().parents[1]/'configs/t07_precision.json').read_text(encoding='utf-8'))


def detect_details(image,bundle=None):
    bundle=bundle or load_bundle()
    original=base_detect(image);height,width=image.shape[:2]
    sigma=estimate_sigma(image);settings=bundle['heads']
    heads,uncertain=refine_heads(image,original['boxes'],settings['level'],settings['sharp_margin'],settings['blur_margin'],settings['tiny_score'])
    calibration=settings['calibration']
    final_heads=[calibrate_head(image,b,sigma,settings['model'],calibration['margin'],calibration['width_scale'],calibration['height_scale']) for b in heads]
    native_plates=[];final_plates=[];rejected=[]
    p=bundle['plates']['blur' if sigma>.9 else 'sharp']
    for box in original['boxes']:
        if box['type']!='plate':continue
        native=profile_panel_bounds(image,box,p['threshold'],0,0,bundle['plates']['require_window'])
        if native is None:rejected.append(dict(box,reason='no_distinct_window_above_body'));continue
        native_plates.append(native)
        if native.get('refinement')=='tiny_occluded_panel_vehicle_fallback':final_plates.append(native)
        else:
            final_plates.append(clip_box(dict(native,x=native['x']-p['left'],y=native['y']-p['top'],
                w=max(1,native['w']+p['left']+p['right']),h=max(1,native['h']+p['top']+p['bottom'])),width,height))
    return dict(boxes=final_heads+final_plates,algorithm=bundle['version'],estimated_sigma=sigma,
        ambiguous_heads=original['ambiguous_heads']+len(uncertain),requires_review=True,
        restoration=original['restoration'],config=bundle,
        stages=dict(proposals=original['boxes'],refined=heads+native_plates,
                    rejected_heads=uncertain,rejected_plates=rejected))


def detect(image,pretrained=()):
    result=detect_details(image)
    return result['boxes'],result['ambiguous_heads']
