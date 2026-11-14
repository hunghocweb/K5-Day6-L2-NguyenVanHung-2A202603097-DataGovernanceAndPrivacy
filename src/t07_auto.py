"""One content-routed detector. Detection and user padding are separate."""
from .t07_router import route_image,RouterConfig,_coverage_preserving_nms
from .t07_detectors import clip_box

VERSION='auto-cascade-1'


class AutoDetectors:
    def __init__(self,params=None,photo_detector=None,drawing_detector=None,router_config=None):
        self.params=params or {};self.photo=photo_detector;self.drawing=drawing_detector
        self.router_config=router_config or RouterConfig()

    def detect_details(self,image,progress=None,routing=None):
        routing=routing or route_image(image,self.router_config);route=routing['route']
        raw=[];merged=[];views=[];ambiguous=0;reasons=[];partial=[]
        cascade=None;selected_algorithm='photo_yunet_yolo';intrinsic_padding=False;padding_policy='explicit_optional_user_padding'
        if route in {'real','both'}:
            if self.photo is None:
                from .t07_multiscale import MultiScaleDetectors
                self.photo=MultiScaleDetectors(self.params)
            trace=self.photo.detect_trace(image,progress)
            raw.extend(trace['raw']);merged.extend(trace['merged']);views=trace['views']
            partial=trace.get('partial_fallbacks',[])
            if partial:reasons.append('crop_edge_partial_detection')
        if route in {'2d','both'}:
            if self.drawing is None:
                from .t07_cascade_2d import Cascade2D
                self.drawing=Cascade2D()
            trace=self.drawing(image);raw.extend(trace['raw']);merged.extend(trace['merged'])
            ambiguous=trace.get('ambiguous_heads',0)
            cascade=trace.get('cascade');selected_algorithm=trace.get('selected_algorithm','injected_2d')
            intrinsic_padding=trace.get('detector_padding_applied',False)
            padding_policy=trace.get('padding_policy',padding_policy)
            if cascade and cascade.get('fallback_executed'):reasons.append(cascade['reason'])
        height,width=image.shape[:2]
        if route=='both':
            merged=_coverage_preserving_nms(merged,width,height,.4);reasons.append('uncertain_image_type')
        else:merged=[clip_box(dict(b),width,height) for b in merged]
        if ambiguous:reasons.append('ambiguous_head')
        if not merged:reasons.append('no_detection_is_not_a_safety_verdict')
        return dict(raw=raw,merged=merged,views=views,routing=routing,algorithm=VERSION,
            ambiguous_heads=ambiguous,review_reasons=reasons,requires_review=True,
            partial_fallbacks=partial,bbox_calibration_applied=False,detector_padding_applied=intrinsic_padding,
            cascade=cascade,selected_algorithm=selected_algorithm,padding_policy=padding_policy)
