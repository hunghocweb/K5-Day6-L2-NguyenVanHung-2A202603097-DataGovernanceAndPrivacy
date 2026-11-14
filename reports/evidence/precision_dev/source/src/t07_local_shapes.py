"""Local shape proposals: edge/MSER windows followed by skin-colour components."""
import cv2
import numpy as np

from .t07_restoration import restore_for_analysis


def head_components(image):
    height,width=image.shape[:2]
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    red,green,blue=(image[:,:,i].astype(np.float32) for i in [2,1,0])
    eligible=((hsv[:,:,0]<=40)&(hsv[:,:,1]>=15)&(red>=10)&(green>red*.30)&(green<red*.99)
              &(blue>green*.18)&(blue<green*.995))
    restored,quality=restore_for_analysis(image,gain=1.2)
    views=[image]+([restored] if quality['applied_sigma'] else [])
    regions=set()
    for view in views:
        gray=cv2.cvtColor(view,cv2.COLOR_BGR2GRAY)
        for low,high in [(2,8),(8,24),(24,60)]:
            contours,_=cv2.findContours(cv2.Canny(gray,low,high),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            regions.update(cv2.boundingRect(c) for c in contours)
        found,_=cv2.MSER_create(2,4,max(100,int(height*width*.2))).detectRegions(gray)
        regions.update(cv2.boundingRect(c) for c in found)
    seen=set()
    for x,y,w,h in sorted(regions):
        if not(3<=w<=width*.95 and 4<=h<=height*.95 and .25<=w/h<=1.35):continue
        margin=max(3,round(max(w,h)*.18))
        xa,ya=max(0,x-margin),max(0,y-margin)
        xb,yb=min(width,x+w+margin),min(height,y+h+margin)
        roi=image[ya:yb,xa:xb];allowed=eligible[ya:yb,xa:xb]
        if int(allowed.sum())<4:continue
        q=roi.astype(np.uint16)//24
        codes=q[:,:,0]+11*q[:,:,1]+121*q[:,:,2]
        values,counts=np.unique(codes[allowed],return_counts=True)
        for index in np.argsort(-counts,kind='stable')[:3]:
            if counts[index]<3:continue
            colour=np.median(roi[(codes==values[index])&allowed],axis=0).astype(np.float32)
            for tolerance in [max(6,float(colour.max())*.12),max(10,float(colour.max())*.24)]:
                field=(np.max(np.abs(roi.astype(np.float32)-colour),axis=2)<=tolerance).astype(np.uint8)
                field=cv2.morphologyEx(field,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
                contours,_=cv2.findContours(field,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    cx,cy,cw,ch=cv2.boundingRect(contour)
                    if cx<=0 or cy<=0 or cx+cw>=roi.shape[1] or cy+ch>=roi.shape[0]:continue
                    if not(4<=cw and 6<=ch and .25<=cw/ch<=1.3):continue
                    gx,gy=xa+cx,ya+cy
                    key=(gx,gy,cw,ch,tuple(int(v)//8 for v in colour))
                    if key in seen:continue
                    seen.add(key)
                    component=field[cy:cy+ch,cx:cx+cw]
                    global_contour=contour+np.array([[[xa,ya]]],np.int32)
                    yield colour,global_contour,component
