"""Synthetic-only geometry and visible-ink detector; no image IDs or review lookup."""
from __future__ import annotations

import cv2
import numpy as np

from .t07_detectors import clip_box, deduplicate, overlap


def facial_features(gray, skin):
    """Require two aligned eyes plus a central lower-face mark (mouth/nose)."""
    h,w=gray.shape
    ys,xs=np.ogrid[:h,:w]
    interior=((xs-(w-1)/2)/max(1,w*.40))**2+((ys-(h-1)/2)/max(1,h*.43))**2<=1
    values=gray[(skin>0)&interior]
    if len(values)<3:return False
    level=float(np.median(values))
    feature=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(max(3,int(w*.55))|1,max(3,int(h*.24))|1)))
    contrast=max(1.5,.02*level)
    dark=(gray<level-max(2,.10*level))&interior
    upper=dark.copy();upper[:max(1,int(h*.22))]=False;upper[max(2,int(h*.62)):]=False
    count,labels,stats,centres=cv2.connectedComponentsWithStats(upper.astype(np.uint8),8)
    eyes=[]
    for i in range(1,count):
        x,y,cw,ch,area=stats[i];cx,cy=centres[i]
        if area>=1 and cw<=w*.26+1 and ch<=h*.23+1:
            if feature[min(h-1,int(cy)),min(w-1,int(cx))]>=contrast:eyes.append((cx,cy))
    pairs=[(a,b) for a in eyes for b in eyes if .12*w<=a[0]<.49*w and .51*w<b[0]<=.88*w
           and .20*w<=b[0]-a[0]<=.72*w and abs(a[1]-b[1])<=max(1,h*.14)]
    # Blurring can join the two eye components. Retain a softer two-sided
    # contrast test, still requiring a lower-face feature and aligned minima.
    regions=[]
    cutoff=level-max(1.2,.04*level)
    ya,yb=max(1,int(h*.27)),max(2,int(h*.59))
    for xa,xb in [(int(w*.23),max(2,int(w*.46))),(int(w*.54),max(3,int(w*.78)))]:
        patch=gray[ya:yb,xa:xb]
        if not patch.size:continue
        ey,ex=np.unravel_index(np.argmin(patch),patch.shape)
        if float(patch[ey,ex])<cutoff and feature[ya+ey,xa+ex]>=contrast:regions.append((xa+ex,ya+ey))
    if not pairs and len(regions)==2 and abs(regions[0][1]-regions[1][1])<=max(1,h*.15):
        pairs=[tuple(regions)]
    if not pairs:return False
    # Rectangular skin regions/back-of-head have no internal facial pattern.
    for a,b in pairs:
        lower=((gray<cutoff)&(feature>=contrast))[max(int(h*.61),int(max(a[1],b[1])+1)):max(2,int(h*.84)),
                   max(1,int(w*.23)):max(2,int(w*.79))]
        if lower.any():return True
    return False


def faces(image):
    original_h,original_w=image.shape[:2]
    scale=2 if max(original_h,original_w)<=1280 else 1
    work=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC) if scale>1 else image
    h,w=work.shape[:2];rgb=work[:,:,::-1].astype(float)
    r,g,b=(rgb[:,:,i] for i in range(3));hsv=cv2.cvtColor(work,cv2.COLOR_BGR2HSV)
    skin=((hsv[:,:,0]<=36)&(hsv[:,:,1]>=22)&(r>=12)&(g>r*.30)&(g<r*.98)
          &(b>g*.18)&(b<g*.99)).astype(np.uint8)*255
    closed=cv2.morphologyEx(skin,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(closed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    gray=cv2.cvtColor(work,cv2.COLOR_BGR2GRAY)
    # MSER recovers a head touching similarly coloured background; no fixed positions.
    regions,_=cv2.MSER_create(4,20,max(50,int(h*w*.08))).detectRegions(gray)
    candidates=[cv2.boundingRect(c) for c in contours]+[cv2.boundingRect(c) for c in regions]
    # A closed ellipse border remains visible when skin touches a building of
    # the same hue; colour connected-components alone merge those two regions.
    for low,high in [(3,10),(10,30),(25,70)]:
        edges=cv2.Canny(gray,low,high)
        rings,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        candidates.extend(cv2.boundingRect(c) for c in rings)
    circles=cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,dp=1.0,minDist=6*scale,
                            param1=45,param2=10*scale,minRadius=3*scale,maxRadius=45*scale)
    if circles is not None:
        for cx,cy,radius in circles[0]:
            x0,y0=max(0,int(cx-radius)),max(0,int(cy-radius*1.25))
            x1,y1=min(w,int(cx+radius+1)),min(h,int(cy+radius*1.25+1))
            candidates.append((x0,y0,x1-x0,y1-y0))
    result=[];ambiguous=0
    for x,y,bw,bh in sorted(set(candidates)):
        if not (4*scale<=bw<=w*.5 and 5*scale<=bh<=h*.6 and .48<=bw/bh<=1.20):continue
        roi_color=work[y:y+bh,x:x+bw].astype(float)
        centre=roi_color[int(bh*.28):max(2,int(bh*.64)),int(bw*.30):max(2,int(bw*.70))]
        if not centre.size:continue
        reference=np.median(centre.reshape(-1,3),axis=0)
        similar=np.max(np.abs(roi_color-reference),axis=2)<=max(12,float(reference.max())*.20)
        roi_skin=closed[y:y+bh,x:x+bw]
        if np.mean(roi_skin>0)>.92:
            roi_skin=(similar&(roi_skin>0)).astype(np.uint8)*255
        occupancy=float(np.mean(roi_skin>0))
        if not .48<=occupancy<=.92:continue
        yy,xx=np.ogrid[:bh,:bw]
        ellipse=(((xx-(bw-1)/2)/max(1,bw/2))**2+((yy-(bh-1)/2)/max(1,bh/2))**2<=1)
        skin_shape=roi_skin>0
        agreement=(skin_shape&ellipse).sum()/max(1,(skin_shape|ellipse).sum())
        if agreement<.85:continue
        if not facial_features(gray[y:y+bh,x:x+bw],skin[y:y+bh,x:x+bw]):
            ambiguous+=1;continue
        result.append(clip_box(dict(x=x/scale-1,y=y/scale-1,w=bw/scale+2,h=bh/scale+2,
            type='face',confidence=float(.75+.15*agreement),source='v3_ellipse_eyes_lower_feature'),original_w,original_h))
    return deduplicate(result,.28),ambiguous


def extend_plate_text(gray,box,color=None):
    """Grow a white substrate to connected dark glyphs on BOTH sides."""
    x,y,w,h=box;image_h,image_w=gray.shape
    partial=w/h<3
    occlusion_gap=0
    if color is not None:
        # A tall, narrow foreground object can interrupt the text string.
        # Permit a larger gap only in that case, rather than jumping to lamps.
        for edge in [x-1,x+w]:
            if not 0<=edge<image_w:continue
            ya,yb=max(0,y-h*5),min(image_h,y+h*3)
            xa,xb=max(0,edge-h*2),min(image_w,edge+h*2+1)
            patch=color[ya:yb,xa:xb].astype(float)
            probe_y=max(0,y-h)
            sample=color[probe_y,edge].astype(float)
            similar=(np.max(np.abs(patch-sample),axis=2)<=22).astype(np.uint8)
            n,labels,stats,centres=cv2.connectedComponentsWithStats(similar,8)
            label=labels[probe_y-ya,edge-xa]
            if label:
                _,_,cw,ch,area=stats[label]
                if ch>=2*h and ch>1.5*cw:
                    partial=True;occlusion_gap=max(occlusion_gap,int(cw)+2)
    reach=max(32 if partial else 20,int(w*.8));x0=max(0,x-reach);x1=min(image_w,x+w+reach)
    margin=max(2,int(h*.75))
    y0=max(0,y-margin);y1=min(image_h,y+h+margin)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(max(11,h*3)|1,max(5,h)|1))
    ink=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,kernel)[y0:y1,x0:x1]
    mask=(ink>=5).astype(np.uint8)
    if color is not None:
        # JPEG blends black letters with the coloured body underneath; don't
        # require the resulting pixel to remain neutral grey.
        mask&=(gray[y0:y1,x0:x1]<145).astype(np.uint8)
    # Remove long car/plate frame edges, keeping compact glyph strokes.
    horizontal=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((1,max(9,h*2)),np.uint8))
    vertical=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((max(9,h*3),1),np.uint8))
    mask=mask&~horizontal&~vertical
    count,labels,stats,centres=cv2.connectedComponentsWithStats(mask,8)
    glyphs=[]
    for i in range(1,count):
        gx,gy,gw,gh,area=map(int,stats[i]);gx+=x0;gy+=y0
        density=area/max(1,gw*gh)
        touches_seed=gx<x+w and gx+gw>x
        narrow_vertical=gw<=2 and gh>=2
        if not touches_seed and not (.35<=density<=.85 or narrow_vertical):continue
        if area>=2 and 1<=gh<=h+2*margin and gw<=w*2.5 and abs(gy+gh/2-(y+h/2))<=h*.5+margin:
            glyphs.append((gx,gy,gw,gh))
    left,right=x,x+w;top,bottom=y,y+h;gap=max(3,int(h*(1.8 if partial else .6)),occlusion_gap)
    # Iteration follows strings of characters, not a fixed width multiplier.
    changed=True
    while changed:
        changed=False
        for gx,gy,gw,gh in glyphs:
            if gx<=right+gap and gx+gw>=left-gap:
                bounds=(min(left,gx),max(right,gx+gw),min(top,gy),max(bottom,gy+gh))
                if bounds!=(left,right,top,bottom):left,right,top,bottom=bounds;changed=True
    return left-1,top-1,right-left+2,bottom-top+2


def plates(image):
    height,width=image.shape[:2];gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);rgb=image[:,:,::-1].astype(float);result=[]
    neutral=(rgb.min(axis=2)>=rgb.max(axis=2)*.8)&(rgb[:,:,2]<=rgb[:,:,0]*1.12)&(rgb[:,:,1]<=rgb[:,:,0]*1.15)
    for level in [90,130,170,210]:
        white=((gray>=level)&(hsv[:,:,1]<=80)&neutral).astype(np.uint8)*255
        white=cv2.morphologyEx(white,cv2.MORPH_CLOSE,np.ones((2,3),np.uint8))
        contours,_=cv2.findContours(white,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x,y,w,h=cv2.boundingRect(contour)
            if not (8<=w<=width*.3 and 3<=h<=height*.10 and 1.5<=w/h<=10):continue
            if cv2.contourArea(contour)/max(1,w*h)<.45:continue
            roi=gray[y:y+h,x:x+w]
            # Letters must be inside the white patch; an empty windshield is rejected.
            inset=max(1,int(h*.18))
            inner=roi[inset:-inset,inset:-inset] if min(w,h)>2*inset else roi
            if inner.size==0 or float(inner.std())<7:continue
            bright=float(np.percentile(roi,80));ink=inner<bright-max(16,bright*.23)
            if ink.size==0 or not .025<=ink.mean()<=.65:continue
            if np.count_nonzero(np.any(ink,axis=0))<max(2,w*.12):continue
            bx,by,bw,bh=extend_plate_text(gray,(x,y,w,h),image)
            result.append(clip_box(dict(x=bx,y=by,w=bw,h=bh,type='plate',confidence=.80,
                                        source='v3_white_substrate_bidirectional_ink'),width,height))
    # Keep the union when multiple brightness thresholds see the same plate.
    merged=[]
    for box in sorted(result,key=lambda b:-b['w']*b['h']):
        for old in merged:
            if overlap(box,old)>.25:
                x0,y0=min(box['x'],old['x']),min(box['y'],old['y'])
                x1,y1=max(box['x']+box['w'],old['x']+old['w']),max(box['y']+box['h'],old['y']+old['h'])
                old.update(x=x0,y=y0,w=x1-x0,h=y1-y0);break
        else:merged.append(box)
    return merged


def detect(image,pretrained=()):
    # Synthetic mode deliberately validates actual image geometry. Real images
    # use the separate pretrained branch; model-only windshield boxes aren't copied.
    face_boxes,ambiguous=faces(image)
    return face_boxes+plates(image),ambiguous
