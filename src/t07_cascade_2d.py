"""Frozen mathematical 2D primary, detailed-drawing mathematical fallback.

The gate measures suitability, never unknown accuracy on an unlabelled image.
No names, labels, saved predictions or photo models enter this module.
"""
from dataclasses import asdict,dataclass
import cv2
import numpy as np

from .t07_router import RouterConfig,_content_roi,_validate_image

VERSION='math2d-cascade-1'


@dataclass(frozen=True)
class GateConfig:
    analysis_side:int=640
    colour_step:int=24
    dominant_colours:int=24
    min_palette_mass:float=.88
    max_edge_fraction:float=.04

    def __post_init__(self):
        if self.analysis_side<32 or not 1<=self.colour_step<=255 or self.dominant_colours<1:
            raise ValueError('Gate image and colour parameters must be positive')
        if not 0<self.min_palette_mass<=1 or not 0<=self.max_edge_fraction<=1:
            raise ValueError('Gate thresholds must be fractions in [0,1]')


def primary_suitability(image,config=None):
    config=config or GateConfig();_validate_image(image)
    roi,crop=_content_roi(image,RouterConfig())
    scale=min(1.,config.analysis_side/max(roi.shape[:2]))
    if scale<1:
        roi=cv2.resize(roi,(max(1,round(roi.shape[1]*scale)),max(1,round(roi.shape[0]*scale))),interpolation=cv2.INTER_AREA)
    quant=roi.astype(np.uint32)//config.colour_step
    bins=256//config.colour_step+1
    code=quant[:,:,0]+bins*quant[:,:,1]+bins*bins*quant[:,:,2]
    _,counts=np.unique(code,return_counts=True)
    mass=float(np.sort(counts)[-config.dominant_colours:].sum()/roi.shape[0]/roi.shape[1])
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    edges=float(np.mean(cv2.Canny(gray,40,100)>0))
    suitable=mass>=config.min_palette_mass and edges<=config.max_edge_fraction
    return dict(suitable=bool(suitable),reason='flat_drawing_supported' if suitable else 'drawing_detail_outside_primary_range',
                palette_mass=mass,edge_fraction=edges,content_roi_xywh=crop,
                feature_shape_hw=list(gray.shape),config=asdict(config),
                measured_accuracy=None,calibration_scope='dev_fixture_and_development_upload_only')


class Cascade2D:
    def __init__(self,primary=None,fallback=None,gate_config=None):
        self.primary=primary;self.fallback=fallback;self.gate_config=gate_config or GateConfig()

    def __call__(self,image):
        gate=primary_suitability(image,self.gate_config)
        primary=None;reason=None
        if gate['suitable']:
            if self.primary is None:
                from .t07_math2d import detect_details
                self.primary=detect_details
            primary=self.primary(image)
            if primary['boxes']:
                boxes=primary['boxes'];raw=[b for group in primary.get('stages',{'final':boxes}).values() for b in group]
                return dict(raw=raw,merged=boxes,views=[],algorithm=VERSION,
                    selected_algorithm=primary.get('algorithm','math2d-dev-1'),
                    ambiguous_heads=primary.get('ambiguous_heads',0),restoration=primary.get('restoration',[]),
                    cascade=dict(primary_executed=True,fallback_executed=False,selected='ellipse_blur_2d',
                                 reason='primary_supported_and_nonempty',gate=gate,primary_count=len(boxes)),
                    bbox_calibration_applied=False,detector_padding_applied=True,
                    padding_policy='frozen_math2d_intrinsic_margins_plus_explicit_optional_user_padding')
            reason='primary_empty_not_a_safety_verdict'
        else:reason=gate['reason']
        if self.fallback is None:
            from .t07_native_2d import detect_details
            self.fallback=detect_details
        result=dict(self.fallback(image))
        result.update(algorithm=VERSION,selected_algorithm='detailed-native-2d-v2',
            cascade=dict(primary_executed=primary is not None,fallback_executed=True,selected='detailed_2d',
                         reason=reason,gate=gate,primary_count=len(primary['boxes']) if primary else None),
            bbox_calibration_applied=False,detector_padding_applied=False,
            padding_policy='native_regions_plus_explicit_optional_user_padding')
        return result
