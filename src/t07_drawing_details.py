"""Pixel proposals for detailed drawings: skin marks and neutral text panels.

No image names, coordinates, review boxes, learned bbox correction or padding.
"""
import cv2
import numpy as np
from .t07_detectors import deduplicate


def _components(mask):
    _,labels,stats,centres=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    return labels,stats[1:],centres[1:]


def text_evidence(image):
    if min(image.shape[:2])<3:return 0,0.
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    neutral=(hsv[:,:,1]<75)&(hsv[:,:,2]>130)
    if neutral.mean()<.30:return 0,0.
    background=float(np.median(gray[neutral]))
    ink=(gray<background-40)&((hsv[:,:,1]<95)|(hsv[:,:,2]<65))
    # The outline of a panel is not lettering.
    _,stats,_=_components(ink)
    h,w=gray.shape;count=0;area=0
    for x,y,bw,bh,n in stats:
        if n<2 or bw>max(3,w*.45) or bh>h*.92:continue
        if not .12*h<=y+bh/2<=.90*h:continue
        if x==0 or y==0 or x+bw>=w or y+bh>=h:continue
        count+=1;area+=int(n)
    return count,area/(w*h)


def plates(image):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);height,width=image.shape[:2]
    result=[]
    for value in [145,190,220]:
        mask=((hsv[:,:,1]<65)&(hsv[:,:,2]>value)).astype(np.uint8)*255
        # Close glyph holes without expanding the outer boundary.
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x,y,w,h=cv2.boundingRect(contour)
            if not (5<=w<=width*.30 and 4<=h<=height*.12 and 1.1<=w/h<=12):continue
            if cv2.contourArea(contour)/(w*h)<.50:continue
            count,fraction=text_evidence(image[y:y+h,x:x+w])
            if count<(1 if w<=14 else 2) or not .03<=fraction<=.5:continue
            result.append(dict(x=x,y=y,w=w,h=h,type='plate',confidence=min(.95,.78+.02*count),
                source='neutral_panel_dark_glyphs',glyph_count=count,ink_fraction=fraction))
    return deduplicate(result,.35)


def faces(image,diagnostics=None,heads=None):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    colour=image.astype(np.float32);b,g,r=cv2.split(colour);height,width=image.shape[:2]
    eligible=(r>g*1.10)&(r>b*.92)&(g>r*.45)&(b>g*.70)&(hsv[:,:,1]>15)&(hsv[:,:,1]<100)&(hsv[:,:,2]>75)&((hsv[:,:,0]<30)|(hsv[:,:,0]>130))
    q=image.astype(np.uint16)//24;codes=q[:,:,0]+11*q[:,:,1]+121*q[:,:,2]
    values,counts=np.unique(codes[eligible],return_counts=True)
    result=[];seen=set()
    for index in np.argsort(-counts,kind='stable')[:128]:
        if counts[index]<8:continue
        reference=np.median(colour[(codes==values[index])&eligible],axis=0)
        mask=(eligible&(np.max(np.abs(colour-reference),axis=2)<=max(10,float(reference.max())*.18))).astype(np.uint8)*255
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x,y,w,h=cv2.boundingRect(contour)
            if not (10<=w<=width*.20 and 10<=h<=height*.25 and .75<=w/h<=3):continue
            if (x,y,w,h) in seen:continue
            seen.add((x,y,w,h))
            polygon=cv2.approxPolyDP(contour,.025*cv2.arcLength(contour,True),True)
            if len(polygon)<5:
                if diagnostics is not None:diagnostics.append(dict(box=[x,y,w,h],rejected='straight_polygon'))
                continue
            patch=gray[y:y+h,x:x+w];part=mask[y:y+h,x:x+w]>0
            if part.mean()<.45:continue
            head=dict(x=x,y=y,w=w,h=h,type='head',confidence=.7,shape='curved_head_like',
                source='local_colour_contour',decision='insufficient_facial_evidence')
            if heads is not None:heads.append(head)
            tone=float(np.median(patch[part]))
            padded=cv2.copyMakeBorder(patch,4,4,4,4,cv2.BORDER_CONSTANT,value=tone)
            response=cv2.morphologyEx(padded,cv2.MORPH_BLACKHAT,np.ones((max(3,round(h*.45)),max(3,round(w*.30))),np.uint8))[4:-4,4:-4]
            ink=(response>max(10,tone*.12)).astype(np.uint8)
            horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,max(5,round(w*.6))),np.uint8))
            vertical=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((max(5,round(h*.7)),1),np.uint8))
            ink&=~(horizontal|vertical)
            mark_labels,stats,centres=_components(ink)
            marks=[]
            for mark_index,((mx,my,mw,mh,area),(cx,cy)) in enumerate(zip(stats,centres),1):
                if area<2 or mw>w*.45 or mh>h*.55:continue
                if (patch[mark_labels==mark_index]<tone*.65).mean()<.35:continue
                if .10*w<=cx<=.90*w and .08*h<=cy<=.90*h:marks.append((cx,cy,int(area)))
            eyes=[(a,b) for a in marks for b in marks if .16*w<=b[0]-a[0]<=.78*w
                  and abs(a[1]-b[1])<=max(1.5,h*.16) and .15*h<min(a[1],b[1]) and max(a[1],b[1])<h*.68]
            frontal=bool(eyes)
            profile=any(.22*h<=b[1]-a[1]<=.65*h and abs(a[0]-b[0])<=w*.4
                        and a[1]<h*.55 for a in marks for b in marks)
            # Dark hair must border the top of the colour component; distant scenery is not enough.
            above=gray[max(0,y-round(h*.45)):min(height,y+max(1,round(h*.12))),x:x+w]
            if diagnostics is not None:diagnostics.append(dict(box=[x,y,w,h],tone=tone,marks=marks,
                eyes=len(eyes),frontal=frontal,profile=profile,hair=float((above<tone*.65).mean()) if above.size else 0))
            if not (frontal or profile):continue
            if not above.size or (above<tone*.65).mean()<.80:continue
            head['decision']='visible_facial_marks'
            result.append(dict(x=x,y=y,w=w,h=h,type='face',confidence=.86,
                source='local_colour_facial_topology',facial_marks=len(marks)))
    return deduplicate(result,.35)
