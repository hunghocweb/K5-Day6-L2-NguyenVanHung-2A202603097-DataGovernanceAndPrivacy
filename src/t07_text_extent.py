"""Trace aligned dark glyphs around a neutral panel without following lamps."""
import cv2
import numpy as np
from .t07_detectors import clip_box


def tiny_panel_bounds(box,sigma,width,height):
    """Conservative bounded fallback when a tiny car reveals only a plate sliver."""
    bx,by,bw,bh=map(float,box['body_xywh'])
    if sigma>.9 and bw<=60 and bh<=25:
        safety=int(np.ceil(sigma))
        return clip_box(dict(box,x=bx-safety,y=by+round(bh*.2),w=bw+2*safety,h=bh+2*safety,
            refinement='tiny_occluded_panel_vehicle_fallback',requires_review=True),width,height)
    return dict(box,refinement='partial_panel_uncertain',requires_review=True)


def paired_window_above_body(image,body,sigma=None):
    x,y,w,h=[int(v) for v in body]
    if sigma is None:
        from .t07_restoration import estimate_sigma
        sigma=estimate_sigma(image)
    # Exclude the Gaussian halo of the candidate's own top edge.
    ya=max(0,round(y-h*.85));yb=min(image.shape[0],y-max(1,int(np.ceil(2*sigma))))
    xa=max(0,x);xb=min(image.shape[1],x+w)
    if yb-ya<3 or xb-xa<8:return False
    patch=cv2.GaussianBlur(image[ya:yb,xa:xb].astype(np.float32),(3,3),.7)
    gradient=np.linalg.norm(np.diff(patch,axis=1),axis=2)
    centre=x+w/2-xa
    l0,l1=max(0,int(centre-.43*w)),max(1,int(centre-.15*w))
    r0,r1=max(1,int(centre+.15*w)),min(gradient.shape[1],int(centre+.43*w))
    if l1<=l0 or r1<=r0:return False
    left,right=gradient[:,l0:l1],gradient[:,r0:r1]
    needed=max(2,round(h*.10))
    for start in range(len(gradient)-needed+1):
        a,b=left[start:start+needed],right[start:start+needed]
        if not ((a.max(axis=1)>3).all() and (b.max(axis=1)>3).all()):continue
        if a.argmax(axis=1).std()<=max(1.5,w*.035) and b.argmax(axis=1).std()<=max(1.5,w*.035):return True
    return False


def glyph_panel_bounds(image,box,threshold=3.,margin_x=1.,margin_y=1.,require_window=False):
    if 'panel_xywh' not in box:return dict(box,refinement='text_no_panel')
    if require_window and ('body_xywh' not in box or not paired_window_above_body(image,box['body_xywh'])):return None
    x,y,w,h=[int(v) for v in box['panel_xywh']]
    height,width=image.shape[:2]
    reach=max(8,round(w*.65));xa=max(0,x-reach);xb=min(width,x+w+reach)
    ya=max(0,y-h);yb=min(height,y+2*h)
    grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)
    vertical_background=cv2.morphologyEx(grey,cv2.MORPH_CLOSE,np.ones((max(5,2*h+1),1),np.uint8))
    response=vertical_background[ya:yb,xa:xb]-grey[ya:yb,xa:xb]
    # Robust high-frequency noise floor from the surrounding original pixels.
    residual=grey[ya:yb,xa:xb]-cv2.GaussianBlur(grey[ya:yb,xa:xb],(3,3),.6)
    noise=float(np.median(np.abs(residual-np.median(residual))))
    ink=(response>max(threshold,3*noise)).astype(np.uint8)
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)[ya:yb,xa:xb]
    local_value=hsv[:,:,2]
    median_value=cv2.medianBlur(local_value,max(5,2*h+1))
    lamp=((hsv[:,:,1]>150)&(local_value.astype(float)>median_value.astype(float)+10)).astype(np.uint8)
    ink&=~cv2.dilate(lamp,np.ones((max(3,h//2),max(3,h//2)),np.uint8))
    _,_,stats,_=cv2.connectedComponentsWithStats(ink,8)
    glyphs=[]
    for gx,gy,gw,gh,area in stats[1:]:
        if gy<=0 or gy+gh>=yb-ya:continue
        gx,gy=int(gx)+xa,int(gy)+ya
        if area<2 or gh>max(4,h*.95) or gw>max(4,h*1.3):continue
        if not y+h*.15<=gy+gh/2<=y+h*.95:continue
        if gh<max(1,h*.1) or area/(gw*gh)<.10:continue
        glyphs.append((gx,gy,int(gw),int(gh)))
    left,right=x,x+w;top,bottom=y,y+h
    gap=max(2,round(h*.45))
    for _ in range(len(glyphs)+1):
        changed=False
        for gx,gy,gw,gh in glyphs:
            if gx<=right+gap and gx+gw>=left-gap:
                bounds=min(left,gx),max(right,gx+gw),min(top,gy),max(bottom,gy+gh)
                if bounds!=(left,right,top,bottom):left,right,top,bottom=bounds;changed=True
        if not changed:break
    return clip_box(dict(box,x=left-margin_x,y=top-margin_y,w=right-left+2*margin_x,
        h=bottom-top+2*margin_y,refinement='vertical_contrast_glyph_chain',glyph_count=len(glyphs),
        noise_floor=noise),width,height)


def profile_panel_bounds(image,box,threshold=.8,margin_x=1.,margin_y=1.,require_window=False):
    """Follow column evidence of vertical local minima, excluding monotone edges."""
    if 'panel_xywh' not in box:return dict(box,refinement='profile_no_panel')
    x,y,w,h=[int(v) for v in box['panel_xywh']];height,width=image.shape[:2]
    if h<4 and 'body_xywh' in box:
        from .t07_restoration import estimate_sigma
        return tiny_panel_bounds(box,estimate_sigma(image),width,height)
    if require_window and ('body_xywh' not in box or not paired_window_above_body(image,box['body_xywh'])):return None
    reach=max(8,round(w*.6));xa=max(0,x-reach);xb=min(width,x+w+reach)
    ya=max(0,y-h);yb=min(height,y+2*h)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)
    patch=cv2.GaussianBlur(gray[ya:yb,xa:xb],(3,3),.5)
    response=np.zeros_like(patch)
    for distance in sorted(set([max(2,round(h*.30)),max(2,round(h*.45)),max(3,round(h*.65))])):
        if 2*distance>=len(patch):continue
        centre=patch[distance:-distance]
        pair=np.minimum(patch[:-2*distance]-centre,patch[2*distance:]-centre)
        response[distance:-distance]=np.maximum(response[distance:-distance],pair)
    hsv=cv2.cvtColor(image[ya:yb,xa:xb],cv2.COLOR_BGR2HSV)
    value=hsv[:,:,2];median=cv2.medianBlur(value,max(5,2*h+1))
    lamp=((hsv[:,:,1]>150)&(value.astype(float)>median.astype(float)+10)).astype(np.uint8)
    response[cv2.dilate(lamp,np.ones((max(3,h//2),max(3,h//2)),np.uint8))>0]=0
    top=max(0,y-ya+round(h*.15));bottom=min(len(response),y-ya+max(2,round(h*.95)))
    profile=response[top:bottom].mean(axis=0)
    # Outside each glyph, broad car edges stay dark at both vertical neighbours
    # and therefore have no local-minimum evidence.
    active=profile>threshold
    n,_,stats,_=cv2.connectedComponentsWithStats(active.reshape(1,-1).astype(np.uint8),8)
    runs=[]
    for px,_,pw,_,_ in stats[1:]:
        intersects=px+xa<x+w and px+xa+pw>x
        if pw>max(w*.8,3*h) and not intersects:continue
        if pw>max(w*1.8,4*h):continue
        runs.append((int(px)+xa,int(pw)))
    left,right=x,x+w;gap=max(2,round(h*.55))
    for _ in range(len(runs)+1):
        changed=False
        for px,pw in runs:
            if px<=right+gap and px+pw>=left-gap:
                a,b=min(left,px),max(right,px+pw)
                if (a,b)!=(left,right):left,right=a,b;changed=True
        if not changed:break
    return clip_box(dict(box,x=left-margin_x,y=y-margin_y,w=right-left+2*margin_x,h=h+2*margin_y,
        refinement='paired_vertical_minima_glyph_profile',profile_max=float(profile.max()),
        trace_runs=runs),width,height)
