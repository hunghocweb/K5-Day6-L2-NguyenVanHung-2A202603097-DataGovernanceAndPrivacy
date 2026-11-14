"""Flat-2D detector: low-contrast facial evidence and bounded text-line growth.

Consumes pixels only. No review labels, image IDs, fixture templates or paths.
Development thresholds must be frozen before evaluating a candidate.
"""
from __future__ import annotations

import cv2
import numpy as np
from functools import lru_cache

from .t07_detectors import clip_box, deduplicate, overlap
from .t07_rules_v3 import facial_features as paired_facial_features


@lru_cache(maxsize=1)
def _facial_templates():
    """Analytic eye/mouth patterns, generated without training images or labels."""
    yy,xx=np.ogrid[:32,:24]
    mask=((xx-11.5)/8.7)**2+((yy-15.5)/11.5)**2<=1
    templates=[];details=[]
    for eye_x in [.28,.33,.38,.43]:
        for eye_y in [.36,.42]:
            for mouth_y in [.66,.73]:
                for sigma in [.7,1.5,2.5,4.,5.5]:
                    template=np.ones((32,24),np.float32)
                    for x in [eye_x,1-eye_x]:
                        cv2.circle(template,(round(23*x),round(31*eye_y)),1,0,-1)
                    cv2.line(template,(9,round(31*mouth_y)),(14,round(31*mouth_y)),0,1)
                    template=cv2.GaussianBlur(template,(0,0),sigma)
                    vector=template[mask];vector=vector-vector.mean()
                    templates.append(vector/max(1e-6,float(np.linalg.norm(vector))))
                    detail=template-cv2.GaussianBlur(template,(0,0),4)
                    vector=detail[mask];vector=vector-vector.mean()
                    details.append(vector/max(1e-6,float(np.linalg.norm(vector))))
    return mask,np.stack(templates),np.stack(details)


def _face_score(gray):
    if min(gray.shape)<4:return 0.
    work=cv2.resize(gray.astype(np.float32),(24,32),interpolation=cv2.INTER_AREA)
    mask,templates,details=_facial_templates()
    vector=work[mask];vector=vector-vector.mean()
    if float(vector.std())<.60:return 0.
    correlation=templates@(vector/max(1e-6,float(np.linalg.norm(vector))))
    highpass=work-cv2.GaussianBlur(work,(0,0),4)
    vector=highpass[mask];vector-=vector.mean()
    high_correlation=details@(vector/max(1e-6,float(np.linalg.norm(vector))))
    return max(float(correlation.max()),float(high_correlation.max())*.85)


@lru_cache(maxsize=1)
def _projected_face_patterns():
    """Project image AND analytic patterns away from the same smooth nuisance field."""
    mask,_,_=_facial_templates()
    yy,xx=np.mgrid[-1:1:32j,-1:1:24j].astype(np.float32)
    design=np.stack([np.ones_like(xx),xx,yy,xx*xx,xx*yy,yy*yy,(xx*xx+yy*yy)**2],axis=-1)[mask]
    basis=np.linalg.qr(design)[0]
    _,templates,_=_facial_templates()
    projected=templates-(templates@basis)@basis.T
    lengths=np.linalg.norm(projected,axis=1)
    projected=projected[lengths>.05]/lengths[lengths>.05,None]
    return mask,basis,projected


@lru_cache(maxsize=1)
def _blurred_head_models():
    """Analytic Gaussian-convolved ellipses and black eye/mouth perturbations."""
    yy,xx=np.mgrid[-1:1:32j,-1:1:24j].astype(np.float32)
    base=np.stack([np.ones_like(xx),xx,yy],axis=-1).reshape(-1,3)
    models=[]
    for extent in [.75,.9,1.05]:
        for sigma in [.5,1.,2.,3.,4.5]:
            canvas=np.zeros((64,56),np.float32)
            cv2.ellipse(canvas,(28,32),(round(11.5*extent),round(15.5*extent)),0,0,360,1,-1)
            ellipse=cv2.GaussianBlur(canvas,(0,0),sigma)[16:48,16:40].ravel()
            design=np.column_stack([base,ellipse])
            basis=np.linalg.qr(design)[0]
            eye_patterns=[];mouth_patterns=[]
            for eye_x in [.28,.34,.40]:
                for eye_y in [.30,.38,.46]:
                    ink=np.zeros((64,56),np.float32)
                    for ex in [eye_x,1-eye_x]:
                        cx=28+round((ex-.5)*23*extent)
                        cy=32+round((eye_y-.5)*31*extent)
                        cv2.circle(ink,(cx,cy),1,1,-1)
                    vector=-cv2.GaussianBlur(ink,(0,0),sigma)[16:48,16:40].ravel()
                    vector=vector-basis@(basis.T@vector)
                    length=float(np.linalg.norm(vector))
                    if length>.03:eye_patterns.append(vector/length)
            for mouth_y in [.58,.66,.74]:
                ink=np.zeros((64,56),np.float32)
                my=32+round((mouth_y-.5)*31*extent)
                cv2.line(ink,(28-round(3*extent),my),(28+round(3*extent),my),1,1)
                vector=-cv2.GaussianBlur(ink,(0,0),sigma)[16:48,16:40].ravel()
                vector=vector-basis@(basis.T@vector)
                length=float(np.linalg.norm(vector))
                if length>.03:mouth_patterns.append(vector/length)
            eyes=np.stack(eye_patterns);mouths=np.stack(mouth_patterns)
            models.append((basis,eyes,mouths,eyes@mouths.T))
    return models


def _head_likelihood(gray):
    """Square-root fraction of residual energy explained by actual dark marks.

    Both hypotheses have a planar background and a blurred solid ellipse.
    Only the face hypothesis may add an eye/mouth pattern. No sample templates
    or fitted image-dependent thresholds are stored in this model bank.
    """
    vector=cv2.resize(gray.astype(np.float32),(24,32),interpolation=cv2.INTER_AREA).ravel()
    null_errors=[];alternative_errors=[]
    for basis,eyes,mouths,cross in _blurred_head_models():
        residual=vector-basis@(basis.T@vector)
        error=float(residual@residual)
        eye_response=eyes@residual;mouth_response=mouths@residual
        er,mr=eye_response[:,None],mouth_response[None,:]
        denominator=np.maximum(1e-5,1-cross*cross)
        eye_coefficient=(er-cross*mr)/denominator
        mouth_coefficient=(mr-cross*er)/denominator
        gain=(er*er+mr*mr-2*cross*er*mr)/denominator
        gain=np.where((eye_coefficient>0)&(mouth_coefficient>=0),gain,0)
        best_gain=max(float(gain.max()),float(np.maximum(eye_response,0).max())**2)
        null_errors.append(error)
        alternative_errors.append(error-best_gain)
    null=min(null_errors)
    if null/len(vector)<.20:return 0.
    gain=max(0.,null-min(alternative_errors))
    return float(np.sqrt(min(1.,gain/max(1e-6,null))))


def _mark_score(gray,skin=None):
    """Partial correlation after removing illumination and radial oval shading.

    This compares a smooth blank-head hypothesis with that same background
    plus Gaussian-blurred eye/mouth patterns. It fits no example-image weights.
    """
    mask,basis,patterns=_projected_face_patterns()
    work=cv2.resize(gray.astype(np.float32),(24,32),interpolation=cv2.INTER_AREA)
    vector=work[mask]
    residual=vector-basis@(basis.T@vector)
    if float(residual.std())<.35:return 0.
    if skin is not None:
        binary=cv2.resize(skin.astype(np.uint8),(24,32),interpolation=cv2.INTER_NEAREST)
        distance=cv2.distanceTransform(binary,cv2.DIST_L2,5)
        if float(np.mean(binary[mask]))<.4 or float(distance.max())<1:return 0.
    return float((patterns@(residual/max(1e-6,float(np.linalg.norm(residual))))).max())


def _internal_marks(gray,skin=None):
    return _mark_score(gray,skin)>=.50


def _local_facials(gray,skin=None):
    """An eye/mouth is darker than BOTH neighbours, unlike an oval's slope."""
    work=cv2.resize(gray.astype(np.float32),(48,64),interpolation=cv2.INTER_LINEAR)
    work=cv2.GaussianBlur(work,(0,0),.6)
    horizontal=np.zeros_like(work);vertical=np.zeros_like(work)
    for step in [3,5,8]:
        horizontal=np.maximum(horizontal,np.minimum(np.roll(work,step,axis=1)-work,np.roll(work,-step,axis=1)-work))
        vertical=np.maximum(vertical,np.minimum(np.roll(work,step,axis=0)-work,np.roll(work,-step,axis=0)-work))
    if skin is not None:
        contours,_=cv2.findContours(skin.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours:return False
        field=np.zeros(skin.shape,np.uint8)
        cv2.fillConvexPoly(field,cv2.convexHull(max(contours,key=cv2.contourArea)),1)
        distance=cv2.distanceTransform(field,cv2.DIST_L2,5)
        valid=cv2.resize((distance>=max(.7,float(distance.max())*.3)).astype(np.uint8),(48,64),interpolation=cv2.INTER_NEAREST)>0
        horizontal=np.where(valid,horizontal,0);vertical=np.where(valid,vertical,0)
    noise=work-cv2.GaussianBlur(work,(0,0),1)
    threshold=max(.65,2.5*float(np.median(np.abs(noise[22:47,12:36]))))
    paired=min(float(horizontal[17:38,10:23].max()),float(horizontal[17:38,25:38].max()))>=threshold
    merged=vertical[17:38,14:34]
    merged_ok=int(np.count_nonzero(np.max(merged,axis=0)>threshold))>=4
    mouth=float(vertical[39:54,14:34].max())>=threshold
    return (paired or merged_ok) and mouth


def _ellipse_bounds(contour,shape):
    """AMS least-squares ellipse; reject unstable extrapolation from short arcs."""
    if len(contour)<5:return None
    try:
        (cx,cy),(major,minor),angle=cv2.fitEllipseAMS(contour)
    except cv2.error:return None
    if not np.isfinite([cx,cy,major,minor,angle]).all() or min(major,minor)<3:return None
    x,y,w,h=cv2.boundingRect(contour)
    if not .35<=min(major,minor)/max(major,minor)<=1:return None
    if not x-w*.25<=cx<=x+w*1.25 or not y-h*.25<=cy<=y+h*1.25:return None
    theta=np.deg2rad(angle);c,s=np.cos(theta),np.sin(theta)
    points=contour.reshape(-1,2).astype(float)-[cx,cy]
    u=points[:,0]*c+points[:,1]*s;v=-points[:,0]*s+points[:,1]*c
    residual=np.abs((u/(major/2))**2+(v/(minor/2))**2-1)
    if float(np.median(residual))>.32:return None
    rx=np.sqrt((major*c/2)**2+(minor*s/2)**2)
    ry=np.sqrt((major*s/2)**2+(minor*c/2)**2)
    x0,y0=max(0,int(np.floor(cx-rx))),max(0,int(np.floor(cy-ry)))
    x1,y1=min(shape[1],int(np.ceil(cx+rx+1))),min(shape[0],int(np.ceil(cy+ry+1)))
    if x1<=x0 or y1<=y0 or (x1-x0)*(y1-y0)>w*h*2:return None
    return x0,y0,x1-x0,y1-y0


def _face_evidence(gray):
    return _face_score(gray)>=.48


def faces(image):
    height, width = image.shape[:2]
    work = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    rgb = work[:, :, ::-1].astype(np.float32)
    r, g, b = (rgb[:, :, i] for i in range(3))
    skin = ((hsv[:, :, 0] <= 38) & (hsv[:, :, 1] >= 18) & (r >= 10)
            & (g > r * .28) & (g < r * .99) & (b > g * .15) & (b < g * .995))
    filled_skin=cv2.morphologyEx(skin.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))>0
    candidates = set()
    # Hue bands split touching heads with different skin colours; no positions.
    masks = [skin]
    for low, high in [(0, 8), (6, 16), (12, 25), (20, 38)]:
        masks.append(skin & (hsv[:, :, 0] >= low) & (hsv[:, :, 0] <= high))
    for low,high in [(18,120),(90,200),(170,255)]:
        masks.append(skin&(hsv[:,:,1]>=low)&(hsv[:,:,1]<=high))
    for mask in masks:
        for size in [1, 3, 5]:
            closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates.update(cv2.boundingRect(c) for c in contours)
            for contour in contours:
                fitted=_ellipse_bounds(contour,gray.shape)
                if fitted is not None:candidates.add(fitted)
        for kernel in [5,9,13]:
            radius=kernel//2
            eroded=cv2.erode(mask.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel,kernel)))
            contours,_=cv2.findContours(eroded,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x,y,w,h=cv2.boundingRect(contour)
                candidates.add((max(0,x-radius),max(0,y-radius),min(work.shape[1]-max(0,x-radius),w+2*radius),min(work.shape[0]-max(0,y-radius),h+2*radius)))
    for low, high in [(2, 7), (8, 20), (20, 60)]:
        contours, _ = cv2.findContours(cv2.Canny(gray, low, high), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates.update(cv2.boundingRect(c) for c in contours)
    for channel in [gray,hsv[:,:,1]]:
        regions, _ = cv2.MSER_create(2, 12, max(100, int(height * width * .12))).detectRegions(channel)
        candidates.update(cv2.boundingRect(c) for c in regions)
    boxes, ambiguous = [], 0
    for x, y, w, h in sorted(candidates):
        if not (10 <= w <= width * .9 and 12 <= h <= height * 1.2 and .48 <= w / h <= 1.25):
            continue
        if float(filled_skin[y:y+h,x:x+w].mean())<.30:continue
        evidence=_face_score(gray[y:y+h,x:x+w])
        if w/h>1.05 and evidence<.68:continue
        roi = filled_skin[y:y+h, x:x+w].copy()
        if float(roi.mean()) > .88 and evidence<.68:
            color = work[y:y+h,x:x+w].astype(np.float32)
            centre = np.median(color[int(h*.25):max(2,int(h*.70)),int(w*.25):max(2,int(w*.75))].reshape(-1,3),axis=0)
            roi &= np.max(np.abs(color-centre),axis=2) <= max(8,float(centre.max())*.18)
        fill = float(roi.mean())
        if not .38 <= fill <= .94 and not (fill>.94 and evidence>=.68):
            continue
        yy, xx = np.ogrid[:h, :w]
        oval = ((xx-(w-1)/2)/max(1,w/2))**2 + ((yy-(h-1)/2)/max(1,h/2))**2 <= 1
        core=((xx-(w-1)/2)/max(1,w*.28))**2+((yy-(h-1)/2)/max(1,h*.32))**2<=1
        if float(roi[core].mean())<.45:continue
        likelihood=_head_likelihood(gray[y:y+h,x:x+w])
        if not _local_facials(gray[y:y+h,x:x+w],roi) and likelihood<.75:continue
        mark_score=_mark_score(gray[y:y+h,x:x+w],roi)
        evidence=max(evidence,mark_score,likelihood)
        agreement = float((roi & oval).sum() / max(1, (roi | oval).sum()))
        paired=agreement>=.80 and paired_facial_features(gray[y:y+h,x:x+w],roi.astype(np.uint8)*255)
        if mark_score<.50 and likelihood<.60 and not paired:continue
        if paired:
            evidence=max(evidence,.68)
        if agreement < .60 or (agreement<.76 and evidence<.68):
            continue
        if evidence<.64:
            ambiguous += 1
            continue
        boxes.append(clip_box(dict(x=x/2-3, y=y/2-3, w=w/2+6, h=h/2+6,
            type="face", confidence=.5+.45*(.8*evidence+.2*agreement), source="v4_local_face_pattern"), width, height))
    # Remove small internal-feature proposals if a larger head encloses them.
    boxes = deduplicate(boxes, .25)
    filtered = []
    for box in boxes:
        inside = any(other is not box and other['w']*other['h'] > 1.8*box['w']*box['h']
                     and other['x'] <= box['x'] and other['y'] <= box['y']
                     and other['x']+other['w'] >= box['x']+box['w']
                     and other['y']+other['h'] >= box['y']+box['h'] for other in boxes)
        if not inside:
            filtered.append(box)
    return filtered, ambiguous


def _plate_ink(gray, box):
    x, y, w, h = box
    roi = gray[y:y+h, x:x+w].astype(np.float32)
    if min(roi.shape) < 3:
        return None
    inset = max(1, round(h*.15))
    inner = roi[inset:h-inset, 1:w-1]
    if inner.size < 4:
        return None
    background = float(np.percentile(roi, 85))
    contrast = background - float(np.percentile(inner, 15))
    if contrast < max(1.5, background*.012):
        return None
    dark = inner < background - max(1.0, contrast*.30)
    if not .04 <= float(dark.mean()) <= .70:
        return None
    profile = (background - inner).mean(axis=0)
    # A blank rectangle only has slow edge shading; text has horizontal variation.
    roughness = float(np.abs(np.diff(profile)).mean()) if len(profile) > 1 else 0
    if roughness < max(.18, contrast*.025):
        return None
    smooth=cv2.GaussianBlur(profile.reshape(1,-1),(0,0),.6).ravel()
    baseline=cv2.GaussianBlur(smooth.reshape(1,-1),(0,0),max(2,w*.13)).ravel()
    detail=smooth-baseline
    peaks=[i for i in range(1,len(detail)-1) if detail[i]>detail[i-1] and detail[i]>=detail[i+1]
           and detail[i]>max(.35,contrast*.025)]
    if len(peaks)<2:
        middle=profile[max(1,int(len(profile)*.3)):max(2,int(len(profile)*.7))]
        shoulders=np.r_[profile[:max(1,int(len(profile)*.2))],profile[-max(1,int(len(profile)*.2)):]]
        if (len(peaks)==0 or not any(.25*len(profile)<=p<=.75*len(profile) for p in peaks)
                or float(middle.mean()-shoulders.mean())<max(1.2,contrast*.10)):
            return None
    if np.count_nonzero(np.any(dark, axis=0)) < max(3, w*.22):
        return None
    # Ink must be concentrated inside the rectangle, not a windshield frame
    # or a tall pedestrian cutting across it from top to bottom.
    row_signal=(background-roi[:,max(1,int(w*.18)):max(2,int(w*.82))]).mean(axis=1)
    edge_signal=float(np.mean(np.r_[row_signal[:1],row_signal[-1:]]))
    centre_signal=float(np.mean(row_signal[max(1,int(h*.25)):max(2,int(h*.75))]))
    return contrast, background, len(peaks)


def extend_plate_text(image, box):
    """Grow only aligned glyphs; reject saturated lamps and tall body/frame edges."""
    x, y, w, h = box
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    margin = max(3, round(h*.70))
    reach = max(8, round(w*.65))
    x0, x1 = max(0,x-reach), min(width,x+w+reach)
    y0, y1 = max(0,y-margin), min(height,y+h+margin)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                              np.ones((max(3,h)|1,max(9,h*2)|1),np.uint8))
    patch = blackhat[y0:y1,x0:x1]
    contrast = max(1., float(np.percentile(patch, 85))*.20)
    value=hsv[:,:,2]
    background_value=cv2.morphologyEx(value,cv2.MORPH_CLOSE,np.ones((max(3,h)|1,max(9,h*2)|1),np.uint8))
    dark_colour=(value[y0:y1,x0:x1]<background_value[y0:y1,x0:x1]*.82)&(hsv[y0:y1,x0:x1,1]<225)
    ink = ((patch >= contrast) & ((hsv[y0:y1,x0:x1,1] < 160)|dark_colour)).astype(np.uint8)
    # A saturated lamp can have dark JPEG rims but a bright coloured core.
    # Remove that core and its rim, rather than following the ring as glyphs.
    local_value=value[y0:y1,x0:x1]
    median_value=cv2.medianBlur(local_value,max(5,h*2)|1)
    lamps=((hsv[y0:y1,x0:x1,1]>160)&(local_value.astype(float)>median_value.astype(float)+12)).astype(np.uint8)
    lamps=cv2.dilate(lamps,np.ones((3,3),np.uint8))
    ink &= ~lamps
    ink &= ~cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,max(9,h*2)),np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    glyphs = []
    for gx,gy,gw,gh,area in stats[1:]:
        gx,gy,gw,gh,area=map(int,(gx+x0,gy+y0,gw,gh,area))
        touches_seed=gx<x+w and gx+gw>x
        if gy<=y0+1 and gy+gh>=y1-1:continue
        if area < 2 or gh > max(h*1.8,h+3) or gw>max(3,w*1.5):continue
        if abs(gy+gh/2-(y+h/2)) > max(2,h*.60):continue
        density=area/(gw*gh)
        if density<.08 or (density>.85 and gw>max(4,3*gh)):continue
        glyphs.append((gx,gy,gw,gh))
    left,right,top,bottom=x,x+w,y,y+h
    gap=max(2,round(h*.55))
    for _ in range(len(glyphs)+1):
        changed=False
        for gx,gy,gw,gh in glyphs:
            if gx <= right+gap and gx+gw >= left-gap:
                new=(min(left,gx),max(right,gx+gw),min(top,gy),max(bottom,gy+gh))
                if new!=(left,right,top,bottom):left,right,top,bottom=new;changed=True
        if not changed:break
    padding=max(5,round(h*.25))
    return left-padding,top-padding,right-left+2*padding,bottom-top+2*padding


def _body_bounds(image,box):
    """Require a compact surrounding colour component for a dark neutral strip.

    The rule distinguishes a car panel from text printed over an unbounded
    sky/road/grass background. Bright standalone plates don't require this.
    """
    x,y,w,h=box;height,width=image.shape[:2]
    xa,xb=max(0,x-3*w),min(width,x+4*w)
    ya,yb=max(0,y-5*h),min(height,y+6*h)
    roi=image[ya:yb,xa:xb]
    ring=np.zeros(roi.shape[:2],bool)
    m=max(2,h//2)
    rx0,rx1=max(0,x-xa-m),min(roi.shape[1],x-xa+w+m)
    ry0,ry1=max(0,y-ya-m),min(roi.shape[0],y-ya+h+m)
    ring[ry0:ry1,rx0:rx1]=True;ring[y-ya:y-ya+h,x-xa:x-xa+w]=False
    if not ring.any():return None
    colour=np.median(roi[ring],axis=0)
    mask=(np.max(np.abs(roi.astype(float)-colour),axis=2)<=max(8,float(colour.max())*.12)).astype(np.uint8)
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    candidates=set(int(v) for v in np.unique(labels[ring]) if v)
    for label in candidates:
        bx,by,bw,bh,area=map(int,stats[label])
        if (bw<1.1*w or bh<1.3*h or not .9<=bw/bh<=6 or area<1.4*w*h):continue
        if bx<=0 or by<=0 or bx+bw>=roi.shape[1] or by+bh>=roi.shape[0]:continue
        return xa+bx,ya+by,bw,bh
    return None


def _bounded_body(image,box):
    return _body_bounds(image,box) is not None


def _vehicle_plate(image,box):
    """A neutral lower panel plus compact body and paired window edges above.

    Used when blur has erased individual letters. Layout is relative to the
    detected body, never to image coordinates, file identity or review labels.
    """
    body=_body_bounds(image,box)
    if body is None:return False
    x,y,w,h=box;bx,by,bw,bh=body
    position=(y+h/2-by)/bh
    if not .40<=position<=1.05 or not 1.2<=bw/bh<=6 or w>bw*.85:return False
    height,width=image.shape[:2]
    ya,yb=max(0,round(by-bh*.7)),min(height,round(by+bh*.22))
    xa,xb=max(0,bx),min(width,bx+bw)
    if yb-ya<3 or xb-xa<8:return False
    patch=image[ya:yb,xa:xb].astype(np.float32)
    patch=cv2.GaussianBlur(patch,(3,3),.7)
    gradient=np.linalg.norm(np.diff(patch,axis=1),axis=2)
    centre=(bx+bw/2)-xa
    left0,left1=max(0,int(centre-.46*bw)),max(1,int(centre-.16*bw))
    right0,right1=max(1,int(centre+.16*bw)),min(gradient.shape[1],int(centre+.46*bw))
    if left1<=left0 or right1<=right0:return False
    left=gradient[:,left0:left1];right=gradient[:,right0:right1]
    threshold=3.
    rows=(left.max(axis=1)>=threshold)&(right.max(axis=1)>=threshold)
    needed=max(2,round(bh*.10))
    for start in range(max(0,len(rows)-needed+1)):
        if not rows[start:start+needed].all():continue
        lx=np.argmax(left[start:start+needed],axis=1)
        rx=np.argmax(right[start:start+needed],axis=1)
        if float(lx.std())<=max(1.5,bw*.035) and float(rx.std())<=max(1.5,bw*.035):return True
    return False


def plates(image):
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bright = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,
                             np.ones((max(15,round(height*.06))|1,max(35,round(width*.12))|1),np.uint8))
    kernel=np.ones((max(15,round(height*.06))|1,max(35,round(width*.12))|1),np.uint8)
    dark=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,kernel)
    neutral_drop=cv2.morphologyEx(hsv[:,:,1],cv2.MORPH_BLACKHAT,kernel)
    seeds=set()
    masks=[]
    for level in [2,4,8,16,32,64]:
        masks.append(((bright>=level)&(hsv[:,:,1]<=65),'bright'))
        masks.append(((dark>=level)&(hsv[:,:,1]<=100)&(neutral_drop>=3),'dark_neutral'))
    for level in [45,75,105,135,165,195,225]:
        masks.append(((gray>=level)&(hsv[:,:,1]<=65),'bright'))
    for mask,kind in masks:
        white=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,_=cv2.findContours(white,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x,y,w,h=cv2.boundingRect(contour)
            if not (7<=w<=width*.8 and 3<=h<=height*.25 and 1.5<=w/h<=12):continue
            if cv2.contourArea(contour)/(w*h)<.48:continue
            margin=max(2,round(h*.6))
            ya,yb=max(0,y-margin),min(height,y+h+margin)
            xa,xb=max(0,x-margin),min(width,x+w+margin)
            neighbourhood=gray[ya:yb,xa:xb]
            ring=np.ones(neighbourhood.shape,bool);ring[y-ya:y-ya+h,x-xa:x-xa+w]=False
            saturation=float(np.median(hsv[y:y+h,x:x+w,1]))
            surround_saturation=float(np.median(hsv[ya:yb,xa:xb,1][ring])) if ring.any() else 0
            if kind=='bright' and saturation>35 and surround_saturation<saturation+20:continue
            if kind=='dark_neutral' and surround_saturation-saturation<max(6,surround_saturation*.15):continue
            substrate=float(np.percentile(gray[y:y+h,x:x+w],60))
            ring_level=float(np.median(neighbourhood[ring])) if ring.any() else 0
            if not ring.any():continue
            if kind=='bright' and substrate-ring_level<max(5,substrate*.05):continue
            if kind=='dark_neutral' and ring_level-float(np.median(gray[y:y+h,x:x+w]))<max(2,ring_level*.02):continue
            if kind=='dark_neutral' and not _bounded_body(image,(x,y,w,h)):continue
            evidence=_plate_ink(gray,(x,y,w,h))
            if evidence is not None or _vehicle_plate(image,(x,y,w,h)):
                seeds.add((x,y,w,h))
    boxes=[]
    # Stronger, compact substrate seeds win; do not union threshold fringe boxes.
    for x,y,w,h in sorted(seeds,key=lambda b:-b[2]*b[3]):
        bx,by,bw,bh=extend_plate_text(image,(x,y,w,h))
        evidence=_plate_ink(gray,(x,y,w,h))
        confidence=.80 if evidence is not None else .65
        boxes.append(clip_box(dict(x=bx,y=by,w=bw,h=bh,type="plate",confidence=confidence,substrate_xywh=[x,y,w,h],
                                  source="v4_local_plate_bounded_glyphs"),width,height))
    boxes=deduplicate(boxes,.25)
    # A window occluded by a pedestrian can imitate one blurred text blob.
    # Prefer a nearby lower text rectangle on the same horizontal car footprint.
    filtered=[]
    for box in boxes:
        has_lower=False
        for other in boxes:
            if other is box:continue
            dy=(other['y']+other['h']/2)-(box['y']+box['h']/2)
            shared=max(0,min(box['x']+box['w'],other['x']+other['w'])-max(box['x'],other['x']))
            if (shared/min(box['w'],other['w'])>=.6 and .7*box['h']<dy<3*box['h']
                    and other['h']<=box['h']*1.2 and box['h']>1.6*other['h']):
                has_lower=True;break
        if not has_lower:filtered.append(box)
    return filtered


def detect(image, pretrained=()):
    """Same interface as v3; caller owns padding, review, export and audit."""
    head_boxes, ambiguous = faces(image)
    return head_boxes + plates(image), ambiguous
