"""Residual head-boundary calibration from size, blur and local contrast."""
import cv2
import numpy as np
from .t07_detectors import clip_box

FEATURES=['bias','inverse_height','blur_per_height','aspect','local_luminance','local_contrast']


def eligible(box):
    return box.get('refinement')=='face_colour_half_level' and min(box['w'],box['h'])>=17


def features(image,box,sigma):
    x,y,w,h=[int(box[k]) for k in ['x','y','w','h']]
    grey=cv2.cvtColor(image[y:y+h,x:x+w],cv2.COLOR_BGR2GRAY).astype(np.float64)
    return np.array([1.,1/h,sigma/h,w/h,float(np.median(grey))/128,float(grey.std())/64])


def target_offsets(box,target):
    x,y,w,h=[float(box[k]) for k in ['x','y','w','h']];gx,gy,gw,gh=[float(target[k]) for k in ['x','y','w','h']]
    return np.array([(gx+gw/2-x-w/2)/h,(gy+gh/2-y-h/2)/h,(gw-w)/h,(gh-h)/h])


def predict(image,box,sigma,model,margin=0.,width_scale=1.,height_scale=1.):
    if not eligible(box):return dict(box)
    x,y,w,h=[float(box[k]) for k in ['x','y','w','h']]
    z=(features(image,box,sigma)-np.asarray(model['mean']))/np.asarray(model['scale'])
    dx,dy,dw,dh=z@np.asarray(model['coefficients'])
    cx=x+w/2+dx*h;cy=y+h/2+dy*h
    rw=np.clip(w+dw*h,.7*w,1.4*w)*width_scale;rh=np.clip(h+dh*h,.7*h,1.4*h)*height_scale
    left,top,right,bottom=np.rint([cx-rw/2-margin,cy-rh/2-margin,cx+rw/2+margin,cy+rh/2+margin])
    return clip_box(dict(box,x=left,y=top,w=max(1,right-left),h=max(1,bottom-top),
        refinement='global_head_boundary_calibration'),image.shape[1],image.shape[0])
