"""Domain-neutral inference: preserve aspect ratio, overlapping crops, class NMS.

Weights are unchanged. This module does not read review labels or synthetic rules.
"""
import cv2
import numpy as np

from .t07_detectors import LocalDetectors, clip_box, deduplicate


def windows(width, height, size=320, overlap=.25):
    def starts(length):
        if length<=size:return [0]
        step=max(1,int(size*(1-overlap)))
        return sorted(set(list(range(0,length-size+1,step))+[length-size]))
    return [(x,y,min(size,width-x),min(size,height-y)) for y in starts(height) for x in starts(width)]


class MultiScaleDetectors(LocalDetectors):
    def detect_view(self,image):
        h,w=image.shape[:2]
        scale=min(320/w,320/h)
        rw,rh=max(1,round(w*scale)),max(1,round(h*scale))
        left,top=(320-rw)//2,(320-rh)//2
        padded=np.full((320,320,3),114,dtype=np.uint8)
        padded[top:top+rh,left:left+rw]=cv2.resize(image,(rw,rh))
        _,faces=self.face.detect(padded)
        boxes=[]
        for face in [] if faces is None else faces:
            x,y,bw,bh=map(float,face[:4])
            if not (left<=x+bw/2<left+rw and top<=y+bh/2<top+rh):continue
            try:
                boxes.append(clip_box(dict(x=(x-left)*w/rw,y=(y-top)*h/rh,w=bw*w/rw,h=bh*h/rh,
                                          type='face',confidence=float(face[-1]),source='yunet_letterbox'),w,h))
            except ValueError:
                continue
        for prediction in self.plate.predict(image):
            b=prediction.bounding_box
            boxes.append(clip_box(dict(x=b.x1,y=b.y1,w=b.width,h=b.height,type='plate',
                                       confidence=float(prediction.confidence),source='yolov9_multiscale'),w,h))
        return boxes

    def detect(self,image):
        return self.detect_trace(image)['merged']

    def detect_trace(self,image, progress=None):
        h,w=image.shape[:2]
        views=[(0,0,w,h)]
        if max(h,w)>320:views+=windows(w,h)
        boxes=[];complete=[];partials=[]
        for index,(x,y,vw,vh) in enumerate(views):
            for box in self.detect_view(image[y:y+vh,x:x+vw]):
                partial=(x,y,vw,vh)!=(0,0,w,h) and (
                    (x>0 and box['x']<=1) or (y>0 and box['y']<=1) or
                    (x+vw<w and box['x']+box['w']>=vw-1) or
                    (y+vh<h and box['y']+box['h']>=vh-1))
                mapped=clip_box({**box,'x':box['x']+x,'y':box['y']+y},w,h)
                boxes.append(mapped)
                (partials if partial else complete).append(mapped)
            if progress:progress(index+1,len(views))
        merged=deduplicate(complete,.35)
        def covered(part):
            for whole in merged:
                if whole['type']!=part['type']:continue
                iw=max(0,min(part['x']+part['w'],whole['x']+whole['w'])-max(part['x'],whole['x']))
                ih=max(0,min(part['y']+part['h'],whole['y']+whole['h'])-max(part['y'],whole['y']))
                if iw*ih/(part['w']*part['h'])>=.8:return True
            return False
        fallback=deduplicate([dict(b,requires_review=True,partial_crop=True) for b in partials if not covered(b)],.35)
        return {'raw':boxes,'merged':merged+fallback,'views':views,'partial_fallbacks':fallback}
