"""2D detections without learned bbox correction or safety padding.

Legacy detectors provide proposals only. Their explicit border padding is
removed here; the caller owns the single user-controlled padding operation.
"""
import cv2
from .t07_detectors import clip_box,deduplicate
from .t07_rules_v3 import faces as coarse_faces
from .t07_rules_v4 import faces as tiny_faces
from .t07_geometry import faces as shape_faces,plates as shape_plates,verify_face_crop
from .t07_restoration import restore_for_analysis
from .t07_precision import remove_fragments
from .t07_face_evidence import joint_facial_score
from .t07_text_extent import profile_panel_bounds
from .t07_drawing_details import faces as detailed_faces,plates as text_plates,text_evidence


def _unpad(box,pixels,width,height):
    if min(box['w'],box['h'])<=2*pixels:return None
    return clip_box(dict(box,x=box['x']+pixels,y=box['y']+pixels,w=box['w']-2*pixels,
                        h=box['h']-2*pixels),width,height)


def detect_details(image):
    height,width=image.shape[:2];raw=detailed_faces(image)+text_plates(image)
    uncertain=[];restoration=[]
    original,ambiguous=coarse_faces(image)
    for gain in [None,.8,1.,1.2]:
        if gain is None:view=image;proposals=original
        else:
            view,quality=restore_for_analysis(image,gain=gain);restoration.append(dict(quality,gain=gain))
            if quality['applied_sigma']==0:continue
            proposals,count=coarse_faces(view);ambiguous=max(ambiguous,count)
        for box in proposals:
            if not verify_face_crop(image,box):continue
            native=_unpad(box,1,width,height)
            if native:raw.append(dict(native,source='native_2d_ellipse'))
    fine,count=shape_faces(image);ambiguous=max(ambiguous,count)
    for box in fine:
        native=_unpad(box,3,width,height)
        if native:raw.append(dict(native,source='native_2d_local_shape'))
    small,count=tiny_faces(image);ambiguous=max(ambiguous,count)
    for box in small:
        if max(box['w'],box['h'])>20:continue
        native=_unpad(box,3,width,height)
        if native is None:continue
        x,y,w,h=[native[k] for k in ['x','y','w','h']]
        if min(w,h)<7:
            uncertain.append(dict(native,reason='too_few_pixels_for_facial_marks'));continue
        score=joint_facial_score(cv2.cvtColor(image[y:y+h,x:x+w],cv2.COLOR_BGR2GRAY))
        if score<.70:uncertain.append(dict(native,reason='weak_eye_mouth_evidence'));continue
        raw.append(dict(native,source='native_2d_tiny_face',joint_score=score))
    for box in shape_plates(image,padding=(0,0),return_proposals=True):
        x,y,w,h=box['panel_xywh']
        glyphs,ink_fraction=text_evidence(image[y:y+h,x:x+w])
        if glyphs<(1 if w<=14 else 2) or ink_fraction<.03:
            uncertain.append(dict(box,reason='panel_without_dark_glyph_evidence'));continue
        if h<4:
            # Visible sliver stays a sliver: no vehicle-wide safety expansion.
            native=clip_box(dict(box,x=x,y=y,w=w,h=h,requires_review=True),width,height)
        else:
            sigma=restoration[0]['estimated_sigma'] if restoration else 0.
            native=profile_panel_bounds(image,box,1. if sigma>.9 else 2.,0.,0.,True)
        if native is not None:raw.append(dict(native,source='native_2d_plate'))
    raw=remove_fragments(raw)
    return dict(raw=raw,merged=deduplicate(raw,.35),views=[],ambiguous_heads=ambiguous+len(uncertain),
                uncertain=uncertain,restoration=restoration,bbox_calibration_applied=False)
