"""Second synthetic-scene experiment; image-derived rules, no per-image lookup.

Designed using dev review feedback. Not a detector for real street imagery.
"""
import cv2
import numpy as np

from .t07_detectors import clip_box, deduplicate, overlap


def faces(image, upscale=True):
    height, width = image.shape[:2]
    rgb = image[:, :, ::-1].astype(np.float32)
    red, green, blue = (rgb[:, :, i] for i in range(3))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Lower value floor retains dark synthetic skin. Relative colour avoids red signs.
    mask = ((hsv[:, :, 0] <= 35) & (hsv[:, :, 1] >= 35) & (hsv[:, :, 1] <= 245)
            & (red >= 15) & (green > red*.33) & (green < red*.96)
            & (blue > green*.22) & (blue < green*.96)).astype(np.uint8)*255
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [cv2.boundingRect(contour) for contour in contours]
    ovals = {cv2.boundingRect(contour) for contour in contours
             if .52 <= cv2.contourArea(contour)/max(1,np.prod(cv2.boundingRect(contour)[2:])) <= .88}
    # Stable grayscale regions recover ovals touching similarly coloured scenery.
    regions, _ = cv2.MSER_create(4, 28, 4200).detectRegions(gray)
    candidates += [cv2.boundingRect(region) for region in regions]
    result, ambiguous = [], 0
    for x,y,w,h in candidates:
        if not (4 <= w <= 65 and 5 <= h <= 80 and .5 <= w/h <= 1.08):
            continue
        roi = gray[y:y+h, x:x+w].astype(float)
        # Contrast relative to surrounding skin, rather than a fixed darkness multiplier.
        # Eyes of dark faces can be almost as bright as their skin.
        skin_values = roi[mask[y:y+h, x:x+w] > 0]
        if not w*h*.5 <= len(skin_values) <= w*h*.92:
            continue
        skin_hue = hsv[y:y+h,x:x+w,0][mask[y:y+h,x:x+w] > 0]
        if np.median(skin_hue) < 5:
            continue
        skin_level = float(np.median(skin_values))
        threshold = skin_level - max(2.5, .12*skin_level)
        ya, yb = int(h*.24), max(int(h*.24)+1, int(h*.59))
        left = roi[ya:yb, max(1,int(w*.18)):max(2,int(w*.46))]
        right = roi[ya:yb, int(w*.54):max(int(w*.54)+1,int(w*.85))]
        evidence = []
        for eye in (left, right):
            ys, xs = np.where(eye < threshold)
            evidence.append(float(np.mean(ys)) if len(xs) >= max(1, int(w*h*.002)) else None)
        paired = all(value is not None for value in evidence) and abs(evidence[0]-evidence[1]) <= max(2,h*.16)
        oval_face = (x,y,w,h) in ovals and .6 <= w/h <= .97
        if paired or oval_face:
            result.append(clip_box(dict(x=x-2, y=y-2, w=w+4, h=h+4, type="face",
                                       confidence=.7, source="v2_skin_eye_contrast"), width, height))
        else:
            ambiguous += 1
    # Pair compact eye-contrast components to split touching heads and find tiny faces.
    contrast = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, np.ones((9,9),np.uint8))
    near_skin = cv2.dilate(mask,np.ones((3,3),np.uint8)) > 0
    eye_mask = ((contrast >= 7) & near_skin).astype(np.uint8)*255
    eye_contours,_ = cv2.findContours(eye_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    eyes=[]
    for contour in eye_contours:
        ex,ey,ew,eh=cv2.boundingRect(contour)
        if 1 <= ew <= 9 and 1 <= eh <= 9 and .4 <= ew/eh <= 2.5:
            eyes.append((ex+ew/2,ey+eh/2,ew*eh))
    for a in eyes:
        for b in eyes:
            distance=b[0]-a[0]
            if not (1.5 <= distance <= 19 and abs(a[1]-b[1]) <= max(1.5,distance*.22)
                    and .25 <= a[2]/b[2] <= 4):
                continue
            fw,fh=distance*2.8,distance*3.5
            x,y=int((a[0]+b[0])/2-fw*.5),int((a[1]+b[1])/2-fh*.39)
            x0,y0=max(0,x),max(0,y)
            x1,y1=min(width,int(x+fw+1)),min(height,int(y+fh+1))
            if x1<=x0 or y1<=y0:continue
            region=mask[y0:y1,x0:x1]>0
            if not .48 <= region.mean() <= .9:continue
            hue=hsv[y0:y1,x0:x1,0][region]
            if np.median(hue)<5:continue
            # Both eyes must be darker than the surrounding skin in this candidate.
            level=float(np.median(gray[y0:y1,x0:x1][region]))
            if any(gray[min(height-1,int(e[1])),min(width-1,int(e[0]))] > level-max(2,.08*level) for e in (a,b)):
                continue
            result.append(clip_box(dict(x=x0-1,y=y0-1,w=x1-x0+2,h=y1-y0+2,type="face",
                                       confidence=.72,source="v2_paired_eye_components"),width,height))
    if upscale:
        enlarged,_=faces(cv2.resize(image,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC),False)
        for b in enlarged:
            if b["h"]<=42:
                result.append(clip_box({**b,"x":b["x"]/2,"y":b["y"]/2,
                                       "w":b["w"]/2,"h":b["h"]/2,
                                       "source":"v2_small_face_upscale"},width,height))
    return deduplicate(result,.35), ambiguous


def plates(image):
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb = image[:, :, ::-1].astype(float)
    lo, hi = rgb.min(axis=2), rgb.max(axis=2)
    # Neutral bright substrates, unlike blue windshields. Closing fills glyph holes.
    contours=[]
    for threshold in (125,155,185,215):
        red,green,blue=rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
        mask = ((gray >= threshold) & (green>=red*.78) & (green<=red*1.12)
                & (blue<=red*1.07) & (blue>=red*.6)).astype(np.uint8)*255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,5),np.uint8))
        found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(found)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT,(11,5)))
    result = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if not (12 <= w <= 100 and 3 <= h <= 24 and 1.7 <= w/h <= 7):
            continue
        interior = blackhat[y:y+h,x:x+w]
        if np.mean(interior) < .35 or np.mean(interior > 2) < .025:
            continue
        # Follow glyph-sized contrast to the right when text overflows its substrate.
        end = min(width,x+w+max(12,int(w*.9)))
        text = (blackhat[y:min(height,y+h),x:end] > 15).astype(np.uint8)
        active = np.where(np.sum(text,axis=0) >= 1)[0]
        right = max(w+3,int(w*1.4))
        previous = max(0,w-3)
        for column in active[active >= w-3]:
            if column-previous > max(4,int(h*.6)):
                break
            right = max(right,int(column)+2)
            previous = int(column)
        pad=5 if h<=10 else 3
        result.append(clip_box(dict(x=x-pad,y=y-pad,w=right+2*pad,h=h+2*pad,type="plate",
                                   confidence=.75,source="v2_neutral_plate_text"),width,height))
    # Merge multiple thresholds for one substrate instead of retaining a small inner box.
    merged=[]
    for box in sorted(result,key=lambda b:-b["w"]*b["h"]):
        for old in merged:
            if overlap(box,old)>.25:
                x0,y0=min(box["x"],old["x"]),min(box["y"],old["y"])
                x1,y1=max(box["x"]+box["w"],old["x"]+old["w"]),max(box["y"]+box["h"],old["y"]+old["h"])
                old.update(x=x0,y=y0,w=x1-x0,h=y1-y0)
                break
        else:merged.append(box)
    return merged


def detect(image, pretrained):
    face_boxes, ambiguous = faces(image)
    plate_boxes = plates(image)
    # Synthetic proposals describe the actual visible substrate/text. Model-only
    # windshield false positives are not carried into this synthetic experiment.
    face_boxes += [b for b in pretrained if b["type"] == "face"]
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    for b in pretrained:
        if b["type"]!="plate":continue
        x,y,w,h=(b[k] for k in ("x","y","w","h"))
        roi=image[y:y+h,x:x+w].astype(float)
        bright=gray[y:y+h,x:x+w]>120
        if bright.sum()<max(3,w*h*.12):continue
        neutral=roi.min(axis=2)>=roi.max(axis=2)*.85
        if (neutral & bright).sum()/bright.sum()<.55:continue
        if any(overlap(b,p)>.15 for p in plate_boxes):continue
        plate_boxes.append(clip_box(dict(x=x-3,y=y-3,w=int(w*1.3)+6,h=h+6,type="plate",
                                        confidence=b["confidence"],source="v2_validated_yolo_plate"),image.shape[1],image.shape[0]))
    return deduplicate(face_boxes + plate_boxes), ambiguous
