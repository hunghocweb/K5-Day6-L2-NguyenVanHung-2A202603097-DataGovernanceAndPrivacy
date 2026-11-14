"""Head precision: original-pixel boundary plus joint tiny-face evidence."""
import cv2
from .t07_precision import PrecisionConfig,refit_face,remove_fragments
from .t07_face_evidence import joint_facial_score
from .t07_detectors import clip_box
from .t07_restoration import estimate_sigma


def refine_heads(image,proposals,level=.5,sharp_margin=.5,blur_margin=2.5,tiny_score=.70):
    sigma=estimate_sigma(image);margin=blur_margin if sigma>.9 else sharp_margin
    height,width=image.shape[:2];heads=[];uncertain=[]
    for box in remove_fragments([b for b in proposals if b['type']=='face']):
        sources=box.get('merged_sources',[box.get('source','')])
        if sources==['v4_local_face_pattern'] and max(box['w'],box['h'])<=20:
            trim=3
            if min(box['w'],box['h'])<=2*trim:uncertain.append(dict(box,reason='tiny_insufficient_resolution'));continue
            core=clip_box(dict(box,x=box['x']+trim,y=box['y']+trim,w=box['w']-2*trim,h=box['h']-2*trim),width,height)
            x,y,w,h=[core[k] for k in ['x','y','w','h']]
            patch=cv2.cvtColor(image[y:y+h,x:x+w],cv2.COLOR_BGR2GRAY)
            score=joint_facial_score(patch)
            if score<tiny_score:uncertain.append(dict(box,reason='tiny_no_joint_eye_mouth_evidence',joint_score=score));continue
            heads.append(clip_box(dict(core,x=x-1,y=y-1,w=w+2,h=h+2,
                refinement='tiny_joint_eye_mouth_and_core',joint_score=score),width,height))
            continue
        fitted=refit_face(image,box,PrecisionConfig(face_margin=0,colour_level=level))
        if fitted.get('refinement')=='face_colour_half_level':
            fitted=clip_box(dict(fitted,x=fitted['x']-margin,y=fitted['y']-margin,
                w=fitted['w']+2*margin,h=fitted['h']+2*margin),width,height)
        heads.append(fitted)
    return heads,uncertain
