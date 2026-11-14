"""Pixel-only mathematical redaction for flat drawings, selected on dev.

No image identifiers, reference labels, saved predictions or network I/O are
accepted here. Keep this branch separate from the pretrained photo detector.
"""
from dataclasses import asdict,dataclass
import cv2
import numpy as np
from .t07_detectors import clip_box
from .t07_fusion import fuse
from .t07_geometry import faces as ellipse_faces,plates as colour_plates,verify_face_crop
from .t07_restoration import restore_for_analysis
from .t07_router import _coverage_preserving_nms
from .t07_rules_v3 import faces as coarse_faces
from .t07_rules_v4 import faces as small_faces

VERSION='math2d-dev-1'


@dataclass(frozen=True)
class Math2DConfig:
    restoration_gains: tuple=(.8,1.,1.2)
    noise_ratio: float=.003
    coarse_padding: float=.01
    small_face_max_size: int=20
    panel_padding_x: int=8
    panel_padding_y: int=4
    fine_face_fraction: float=0.
    plate_support_fraction: float=0.


def detect_details(image,config=None):
    config=config or Math2DConfig()
    if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or min(image.shape[:2])<2:
        raise ValueError('Expected a nonempty uint8 BGR image')
    height,width=image.shape[:2]
    raw,ambiguous=coarse_faces(image)
    coarse=list(raw);restoration=[]
    for gain in config.restoration_gains:
        restored,quality=restore_for_analysis(image,gain,config.noise_ratio)
        quality['gain']=gain;restoration.append(quality)
        if quality['applied_sigma']==0:
            proposals=raw
        else:
            proposals,count=coarse_faces(restored)
            ambiguous=max(ambiguous,count)
        coarse.extend(clip_box(b,width,height,config.coarse_padding) for b in proposals)
    coarse=_coverage_preserving_nms(coarse,width,height,.25)
    # Restored views may ring around a blank head. Evidence must also exist
    # in the original pixels; restored texture alone cannot authorize a face.
    verified=[b for b in coarse if verify_face_crop(image,b)]
    if len(verified)<len(coarse):ambiguous=max(ambiguous,1)
    coarse=verified
    fine,unresolved=ellipse_faces(image)
    fine+=colour_plates(image,padding=(config.panel_padding_x,config.panel_padding_y))
    small,count=small_faces(image)
    small=[b for b in small if max(b['w'],b['h'])<=config.small_face_max_size]
    boxes=fuse(coarse,fine,small,width,height,config.fine_face_fraction,config.plate_support_fraction)
    return dict(boxes=boxes,ambiguous_heads=max(ambiguous,unresolved,count),
        algorithm=VERSION,config=asdict(config),restoration=restoration,
        stages=dict(coarse=coarse,fine=fine,small=small),requires_review=True)


def detect(image,pretrained=()):
    result=detect_details(image)
    return result['boxes'],result['ambiguous_heads']
