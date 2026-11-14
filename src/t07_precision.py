"""Refit redaction boundaries from original pixels, independently of labels.

Input rectangles locate objects; they do not determine the final margin.
Only this module is experimental. The frozen detector remains unchanged.
"""
from dataclasses import dataclass,asdict
import cv2
import numpy as np
from .t07_detectors import clip_box,overlap
from .t07_restoration import estimate_sigma


@dataclass(frozen=True)
class PrecisionConfig:
    face_margin: float=1.
    plate_margin_x: float=2.
    plate_margin_y: float=1.
    colour_level: float=.5
    remove_contained: bool=True


def remove_fragments(boxes):
    """Drop enclosed same-class fragments without averaging them into a head."""
    kept=[]
    for b in boxes:
        if any(o is not b and o['type']==b['type'] and o['w']*o['h']>4*b['w']*b['h']
            and o['w']>1.5*b['w'] and o['h']>1.5*b['h']
            and o['x']<=b['x'] and o['y']<=b['y']
            and o['x']+o['w']>=b['x']+b['w'] and o['y']+o['h']>=b['y']+b['h'] for o in boxes):continue
        kept.append(dict(b))
    return kept


def _roi(image,box,margin):
    height,width=image.shape[:2]
    x0=max(0,int(np.floor(box['x']-margin)));y0=max(0,int(np.floor(box['y']-margin)))
    x1=min(width,int(np.ceil(box['x']+box['w']+margin)));y1=min(height,int(np.ceil(box['y']+box['h']+margin)))
    return image[y0:y1,x0:x1],x0,y0


def refit_face(image,box,config):
    """Locate the 50% colour mixture contour instead of a padded proposal."""
    patch,x0,y0=_roi(image,box,max(4,round(max(box['w'],box['h'])*.22)))
    colour=patch.astype(np.float32)
    cy=box['y']+box['h']/2-y0;cx=box['x']+box['w']/2-x0
    yy,xx=np.mgrid[:patch.shape[0],:patch.shape[1]]
    radius=((xx-cx)/max(2,box['w']/2-2))**2+((yy-cy)/max(2,box['h']/2-2))**2
    r,g,b=colour[:,:,2],colour[:,:,1],colour[:,:,0]
    skin=(r>g*1.01)&(g>r*.30)&(g>b*1.005)&(b>g*.18)
    core=(radius<.60)&skin
    outer=(radius>1.15)&(radius<1.9)&(yy<cy+box['h']*.32)
    if int(core.sum())<4 or int(outer.sum())<8:return dict(box,refinement='face_insufficient_support')
    foreground=np.median(colour[core],axis=0)
    background=np.median(colour[outer],axis=0)
    direction=foreground-background;power=float(direction@direction)
    if power<36:return dict(box,refinement='face_low_colour_contrast')
    alpha=((colour-background)@direction)/power
    field=((alpha>=config.colour_level)&skin).astype(np.uint8)
    field=cv2.morphologyEx(field,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(field,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    choices=[]
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        if x==0 or y==0 or x+w>=patch.shape[1] or y+h>=patch.shape[0]:continue
        if not (.45*box['w']<=w<=1.2*box['w'] and .5*box['h']<=h<=1.2*box['h']):continue
        proposal=dict(box,x=x+x0,y=y+y0,w=w,h=h)
        if overlap(box,proposal)<.25 or len(contour)<5:continue
        ellipse=cv2.fitEllipseAMS(contour)
        oval=np.zeros(field.shape,np.uint8);cv2.ellipse(oval,ellipse,1,-1)
        component=np.zeros_like(field);cv2.drawContours(component,[contour],-1,1,-1)
        agreement=float(((oval>0)&(component>0)).sum()/max(1,((oval>0)|(component>0)).sum()))
        if agreement<.75:continue
        choices.append((overlap(box,proposal)*agreement,proposal,agreement))
    if not choices:return dict(box,refinement='face_no_closed_ellipse')
    _,proposal,agreement=max(choices,key=lambda item:item[0])
    margin=config.face_margin
    return clip_box(dict(proposal,x=proposal['x']-margin,y=proposal['y']-margin,
        w=proposal['w']+2*margin,h=proposal['h']+2*margin,refinement='face_colour_half_level',
        boundary_agreement=agreement),image.shape[1],image.shape[0])


def refit_plate(image,box,config):
    """Neutral foreground removes body chroma even when letters are blurred."""
    if 'body_xywh' not in box or 'panel_xywh' not in box:return dict(box,refinement='plate_no_context')
    height,width=image.shape[:2]
    bx,by,bw,bh=[int(v) for v in box['body_xywh']]
    body=image[by:by+bh,bx:bx+bw].astype(np.float32)
    if not body.size:return dict(box,refinement='plate_no_body')
    colour=np.median(body.reshape(-1,3),axis=0)
    direction=np.array([colour[2]-colour[1],colour[0]-colour[1]],np.float32)
    power=float(direction@direction)
    if power<64:return dict(box,refinement='plate_low_chroma')
    x,y,w,h=[int(v) for v in box['panel_xywh']]
    xa=max(bx,x-max(5,round(h)));xb=min(bx+bw,x+w+max(5,round(h)))
    ya=max(by,y-max(3,round(h*.5)));yb=min(by+bh,y+h+max(3,round(h*.5)))
    patch=image[ya:yb,xa:xb].astype(np.float32)
    if min(patch.shape[:2])<2:return dict(box,refinement='plate_empty_patch')
    opponent=np.stack([patch[:,:,2]-patch[:,:,1],patch[:,:,0]-patch[:,:,1]],axis=-1)
    factor=(opponent@direction)/power
    error=np.linalg.norm(opponent-factor[:,:,None]*direction,axis=2)
    alpha=1-factor
    field=((alpha>=config.colour_level)&(error<max(8,np.sqrt(power)*.35))).astype(np.uint8)
    field=cv2.morphologyEx(field,cv2.MORPH_CLOSE,np.ones((1,3),np.uint8))
    contours,_=cv2.findContours(field,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    choices=[]
    for contour in contours:
        px,py,pw,ph=cv2.boundingRect(contour)
        if pw<5 or ph<2 or not 1.3<=pw/ph<=16:continue
        if not .25*w<=pw<=1.6*w or ph>1.5*h+2:continue
        density=float(field[py:py+ph,px:px+pw].mean())
        if density<.62:continue
        candidate=dict(box,x=px+xa,y=py+ya,w=pw,h=ph)
        proximity=overlap(candidate,dict(box,x=x,y=y,w=w,h=h))
        if proximity<.15:continue
        choices.append((proximity*density,candidate,density))
    if not choices:return dict(box,refinement='plate_no_filled_panel')
    _,candidate,density=max(choices,key=lambda item:item[0])
    dx,dy=config.plate_margin_x,config.plate_margin_y
    return clip_box(dict(candidate,x=candidate['x']-dx,y=candidate['y']-dy,w=candidate['w']+2*dx,
        h=candidate['h']+2*dy,refinement='plate_neutral_half_level',boundary_density=density),width,height)


def refine_detections(image,boxes,config=None):
    config=config or PrecisionConfig()
    candidates=remove_fragments(boxes) if config.remove_contained else [dict(b) for b in boxes]
    result=[refit_face(image,b,config) if b['type']=='face' else refit_plate(image,b,config) for b in candidates]
    return result,dict(config=asdict(config),input_count=len(boxes),output_count=len(result))
