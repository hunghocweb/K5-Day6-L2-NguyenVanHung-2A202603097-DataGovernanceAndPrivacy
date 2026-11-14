"""Colour components and enclosed neutral panels for flat 2D vehicle drawings.

All coordinates are measured from pixels. A car-like compact colour component
with paired window edges provides context when blur erases plate lettering.
"""
import cv2
import numpy as np

from .t07_detectors import clip_box, deduplicate
from .t07_rules_v3 import facial_features as paired_facial_features


def complete_ellipse(contour):
    """RANSAC on an observed arc, treating a straight occluder as outliers."""
    x,y,w,h=cv2.boundingRect(contour)
    if w/h>.68:return None
    raster=np.zeros((h,w),np.uint8)
    cv2.drawContours(raster,[contour-np.array([[[x,y]]],np.int32)],-1,1,-1)
    outlines,_=cv2.findContours(raster,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    points=max(outlines,key=cv2.contourArea).reshape(-1,2).astype(np.float32)+[x,y]
    if len(points)<6:return None
    before,after=np.roll(points,3,axis=0),np.roll(points,-3,axis=0)
    chord=after-before
    distance=np.abs(chord[:,0]*(before[:,1]-points[:,1])-chord[:,1]*(before[:,0]-points[:,0]))/np.maximum(1,np.linalg.norm(chord,axis=1))
    points=points[(distance>.12)&(distance<1.5)].astype(np.float32)
    if len(points)<6:return None
    rng=np.random.default_rng(42);best=None;best_score=-1;best_points=None
    for _ in range(48):
        sample=points[rng.choice(len(points),size=min(7,len(points)),replace=False)].reshape(-1,1,2)
        try:ellipse=cv2.fitEllipseAMS(sample)
        except cv2.error:continue
        (cx,cy),(a,b),angle=ellipse
        if not np.isfinite([cx,cy,a,b,angle]).all() or min(a,b)<3:continue
        if not .60<=min(a,b)/max(a,b)<=1:continue
        theta=np.deg2rad(angle);c,s=np.cos(theta),np.sin(theta)
        rx=np.sqrt((a*c/2)**2+(b*s/2)**2);ry=np.sqrt((a*s/2)**2+(b*c/2)**2)
        if not 1.15*w<=2*rx<=3*w or not .8*h<=2*ry<=1.4*h:continue
        if not x-w*.5<=cx<=x+w*1.5 or not y<=cy<=y+h:continue
        u=(points[:,0]-cx)*c+(points[:,1]-cy)*s
        v=-(points[:,0]-cx)*s+(points[:,1]-cy)*c
        radius=np.sqrt((u/(a/2))**2+(v/(b/2))**2)
        error=np.abs(radius-1)*min(a,b)/2
        inliers=error<=max(1.2,h*.04)
        if float(inliers.mean())<.55:continue
        angles=np.sort(np.mod(np.arctan2(v[inliers]/b,u[inliers]/a),2*np.pi))
        if len(angles)<5:continue
        span=2*np.pi-float(np.diff(np.r_[angles,angles[0]+2*np.pi]).max())
        if span<2.1:continue
        score=int(inliers.sum())-.1*float(error[inliers].sum())-.5*abs(min(a,b)/max(a,b)-.75)
        if score>best_score:best_score=score;best=ellipse;best_points=points[inliers]
    if best_points is not None and len(best_points)>=5:
        try:
            refined=cv2.fitEllipseAMS(best_points.reshape(-1,1,2))
            (cx,cy),(a,b),_=refined
            if np.isfinite([cx,cy,a,b]).all() and .60<=min(a,b)/max(a,b)<=1:best=refined
        except cv2.error:pass
    return best


def facial_marks(channel,luma=None):
    """Internal topological dark marks, excluding the ellipse boundary."""
    channel=np.asarray(channel,np.float32)
    blackhat=cv2.morphologyEx(channel,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,7)))
    yy,xx=np.ogrid[:32,:24]
    core=((xx-12)/8)**2+((yy-16)/11)**2<=1
    noise=channel-cv2.GaussianBlur(channel,(0,0),.8)
    cheek=np.r_[noise[17:22,5:9].ravel(),noise[17:22,16:20].ravel()]
    threshold=max(5.,4*float(np.median(np.abs(cheek))))
    if luma is not None:
        luma=np.asarray(luma,np.float32)
        luma_hat=cv2.morphologyEx(luma,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,7)))
        luma_noise=luma-cv2.GaussianBlur(luma,(0,0),.8)
        luma_cheek=np.r_[luma_noise[17:22,5:9].ravel(),luma_noise[17:22,16:20].ravel()]
        luma_threshold=max(1.2,3*float(np.median(np.abs(luma_cheek))))
        blackhat=np.minimum(blackhat/threshold,luma_hat/luma_threshold)
        threshold=1.
    binary=((blackhat>=threshold*.65)&core).astype(np.uint8)
    count,labels,stats,centres=cv2.connectedComponentsWithStats(binary,8)
    eyes=[];mouths=[]
    for index in range(1,count):
        x,y,w,h,area=map(int,stats[index]);cx,cy=centres[index]
        peak=float(blackhat[labels==index].max())
        if area<2:continue
        if 6<=cx<=18 and 7<=cy<=18 and w<=15 and h<=12 and peak>=threshold:eyes.append((index,cx,cy))
        if 7<=cx<=17 and 19<=cy<=27 and w<=15 and h<=9 and peak>=threshold*.7:mouths.append((index,cx,cy))
    if any(a[0]!=b[0] and b[2]-a[2]>=4 for a in eyes for b in mouths):return True
    if any(a[0]!=b[0] and 4<=b[1]-a[1]<=13 and abs(a[2]-b[2])<=3 for a in eyes for b in eyes):return True
    # Gaussian tails may connect eyes and mouth without removing their two peaks.
    profile=np.max(np.where(core,blackhat,0)[:,6:19],axis=1)
    upper=7+int(np.argmax(profile[7:18]));lower=19+int(np.argmax(profile[19:27]))
    if profile[upper]>=threshold and profile[lower]>=threshold*.7:
        if lower-upper>=5 and float(profile[upper+1:lower].min())<min(profile[upper],profile[lower])*.70:return True
    return False


def tiny_face_bands(channel,luma=None):
    """At low resolution, test two dark row bands instead of separated pupils."""
    def response(image):
        profile=np.mean(np.asarray(image,np.float32)[:,6:19],axis=1)
        profile=cv2.GaussianBlur(profile.reshape(-1,1),(0,0),.6).ravel()
        detail=profile-cv2.GaussianBlur(profile.reshape(-1,1),(0,0),1.).ravel()
        threshold=max(.7,3*float(np.median(np.abs(detail[7:28]))))
        evidence=np.zeros_like(profile)
        for step in [4,6]:
            evidence=np.maximum(evidence,np.minimum(np.roll(profile,step)-profile,np.roll(profile,-step)-profile))
        return evidence/threshold
    evidence=response(channel)
    if luma is not None:evidence=np.minimum(evidence,response(luma))
    upper=7+int(np.argmax(evidence[7:18]));lower=19+int(np.argmax(evidence[19:27]))
    return lower-upper>=5 and evidence[upper]>=1 and evidence[lower]>=.7


def window_support(image,body):
    x,y,w,h=body;height,width=image.shape[:2]
    ya,yb=max(0,round(y-h*.7)),min(height,round(y+h*.22))
    xa,xb=max(0,x),min(width,x+w)
    if yb-ya<3 or xb-xa<8:return False
    patch=cv2.GaussianBlur(image[ya:yb,xa:xb].astype(np.float32),(3,3),.7)
    gradient=np.linalg.norm(np.diff(patch,axis=1),axis=2)
    centre=(x+w/2)-xa
    l0,l1=max(0,int(centre-.40*w)),max(1,int(centre-.20*w))
    r0,r1=max(1,int(centre+.20*w)),min(gradient.shape[1],int(centre+.40*w))
    if l1<=l0 or r1<=r0:return False
    left,right=gradient[:,l0:l1],gradient[:,r0:r1]
    rows=(left.max(axis=1)>3)|(right.max(axis=1)>3)
    needed=max(2,round(h*.10))
    for start in range(max(0,len(rows)-needed+1)):
        if rows[start:start+needed].all():
            left_ok=(left[start:start+needed].max(axis=1)>3).all() and np.argmax(left[start:start+needed],axis=1).std()<=max(1.5,w*.035)
            right_ok=(right[start:start+needed].max(axis=1)>3).all() and np.argmax(right[start:start+needed],axis=1).std()<=max(1.5,w*.035)
            if left_ok or right_ok:return True
    return False


def colour_bodies(image):
    height,width=image.shape[:2]
    smooth=cv2.pyrMeanShiftFiltering(image,3,12,maxLevel=1)
    quant=smooth.astype(np.uint16)//24
    codes=quant[:,:,0]+11*quant[:,:,1]+121*quant[:,:,2]
    values,counts=np.unique(codes,return_counts=True)
    order=np.argsort(-counts,kind='stable')[:64]
    seen=set()
    for i in order:
        if counts[i]<10:continue
        pixels=smooth[codes==values[i]]
        colour=np.median(pixels,axis=0)
        for tolerance in [6,12]:
            mask=(np.max(np.abs(smooth.astype(np.float32)-colour),axis=2)<=tolerance).astype(np.uint8)
            mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x,y,w,h=cv2.boundingRect(contour)
                if not (14<=w<=width*.9 and 7<=h<=height*.4 and 1.5<=w/h<=7):continue
                if x<=1 or y<=1 or x+w>=width-1 or y+h>=height-1:continue
                if cv2.contourArea(contour)/(w*h)<.45:continue
                key=(x,y,w,h,tuple(int(c) for c in colour))
                if key in seen:continue
                seen.add(key)
                if not window_support(image,(x,y,w,h)):continue
                yield (x,y,w,h),colour,mask,contour


def plates(image, padding=(4,4), return_proposals=False):
    height,width=image.shape[:2]
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    proposals=[]
    rgb=image[:,:,::-1].astype(np.float32)
    opponent=np.stack([rgb[:,:,0]-rgb[:,:,1],rgb[:,:,2]-rgb[:,:,1]],axis=-1)
    for (x,y,w,h),colour,mask,contour in colour_bodies(image):
        hull=cv2.convexHull(contour)
        canvas=np.zeros(mask.shape,np.uint8);cv2.fillConvexPoly(canvas,hull,1)
        inside=cv2.distanceTransform(canvas,cv2.DIST_L2,5)>=max(1.5,h*.06)
        body_hsv=cv2.cvtColor(np.uint8([[colour]]),cv2.COLOR_BGR2HSV)[0,0]
        chroma=np.array([colour[2]-colour[1],colour[0]-colour[1]],np.float32)
        power=float(chroma@chroma)
        if power>=64:
            factor=(opponent@chroma)/power
            perpendicular=np.linalg.norm(opponent-factor[:,:,None]*chroma,axis=2)
            holes=(inside&(factor<.80)&(factor>=-.20)&(perpendicular<max(5,np.sqrt(power)*.30))).astype(np.uint8)
        else:
            neutral=hsv[:,:,1]<=max(40,float(body_hsv[1])*.95)
            holes=(inside&(mask==0)&neutral).astype(np.uint8)
        holes=cv2.morphologyEx(holes,cv2.MORPH_CLOSE,np.ones((1,3),np.uint8))
        contours,_=cv2.findContours(holes[y:y+h,x:x+w],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        parts=[]
        for part in contours:
            px,py,pw,ph=cv2.boundingRect(part);px+=x;py+=y
            position=(py+ph/2-y)/h
            if not .40<=position<=.98 or ph>h*.65 or pw>w*.85:continue
            if abs(px+pw/2-(x+w/2))>w*.25:continue
            if pw*ph<4:continue
            parts.append([px,py,pw,ph])
        # Join aligned neutral glyph fragments only within the same car body.
        parts.sort(key=lambda b:b[0])
        lines=[]
        for part in parts:
            px,py,pw,ph=part
            for line in lines:
                lx,ly,lw,lh=line
                if abs(py+ph/2-(ly+lh/2))<=max(2,max(ph,lh)*.45) and px-(lx+lw)<=max(3,max(ph,lh)*.8):
                    x0,y0=min(lx,px),min(ly,py);x1,y1=max(lx+lw,px+pw),max(ly+lh,py+ph)
                    line[:]=[x0,y0,x1-x0,y1-y0];break
            else:lines.append(part.copy())
        for px,py,pw,ph in lines:
            if pw<5 or ph<2 or not 1.3<=pw/ph<=14:continue
            dx=max(padding[0],round(ph*.25));dy=max(padding[1],round(ph*.25))
            proposals.append(clip_box(dict(x=px-dx,y=py-dy,w=pw+2*dx,h=ph+2*dy,
                type='plate',confidence=.72,source='colour_body_neutral_panel_geometry',
                body_xywh=[x,y,w,h],panel_xywh=[px,py,pw,ph]),width,height))
    proposals.sort(key=lambda b:-b['w']*b['h'])
    if return_proposals:return proposals
    return deduplicate(proposals,.30)


def verify_face_crop(image,box):
    """Require facial marks beyond the shading of a solid blurred head."""
    from .t07_rules_v4 import _head_likelihood,_mark_score
    x,y,w,h=[int(box[k]) for k in ['x','y','w','h']]
    gray=cv2.cvtColor(image[y:y+h,x:x+w],cv2.COLOR_BGR2GRAY)
    return _head_likelihood(gray)>=.15 and _mark_score(gray)>=.30


def faces(image,likelihood=None):
    """Local component ellipses verified by aligned luminance/chroma marks."""
    from .t07_local_shapes import head_components
    height,width=image.shape[:2]
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    boxes=[];seen=set();ambiguous=0
    for colour,contour,component in head_components(image):
        x,y,w,h=cv2.boundingRect(contour)
        completed=complete_ellipse(contour)
        if completed is not None:
            (cx,cy),(a,b),angle=completed
        elif len(contour)<5 or (w<12 and h<16):
            moments=cv2.moments(component)
            if moments['m00']<4:continue
            centre=np.array([moments['m10'],moments['m01']])/moments['m00']
            covariance=np.array([[moments['mu20'],moments['mu11']],[moments['mu11'],moments['mu02']]])/moments['m00']
            eigenvalues,vectors=np.linalg.eigh(covariance)
            if min(eigenvalues)<=0:continue
            a,b=4*np.sqrt(eigenvalues);cx,cy=centre+[x,y]
            angle=float(np.rad2deg(np.arctan2(vectors[1,0],vectors[0,0])))
        else:
            try:(cx,cy),(a,b),angle=cv2.fitEllipseAMS(contour)
            except cv2.error:continue
        if not np.isfinite([cx,cy,a,b,angle]).all() or min(a,b)<3:continue
        if not .40<=min(a,b)/max(a,b)<=1 or a*b>w*h*(3 if completed is not None else 2):continue
        if not x-w*.5<=cx<x+w*1.5 or not y<=cy<y+h:continue
        yy,xx=np.mgrid[y:y+h,x:x+w].astype(np.float32)
        t=np.deg2rad(angle);ct,st=np.cos(t),np.sin(t)
        u=(xx-cx)*ct+(yy-cy)*st;v=-(xx-cx)*st+(yy-cy)*ct
        core=(u/max(1,a*.32))**2+(v/max(1,b*.32))**2<=1
        if not core.any() or float(component[core].mean())<.45:continue
        margin=max(3,int(np.ceil(max(a,b)*.5)))
        field=np.zeros((h+2*margin,w+2*margin),np.uint8)
        shifted=contour-np.array([[[x-margin,y-margin]]],np.int32)
        cv2.drawContours(field,[shifted],-1,1,-1)
        oval=np.zeros_like(field)
        cv2.ellipse(oval,((cx-x+margin,cy-y+margin),(a,b),angle),1,-1)
        agreement=float(((oval>0)&(field>0)).sum()/max(1,((oval>0)|(field>0)).sum()))
        if agreement<(.35 if completed is not None else .78):continue
        key=(round(cx),round(cy),round(a),round(b),round(angle))
        if key in seen:continue
        seen.add(key)
        if a>b:a,b=b,a;angle+=90
        if a/b>.9:angle=0
        theta=np.deg2rad(angle);c,s=np.cos(theta),np.sin(theta)
        sx,sy=23/max(1,a*1.15),31/max(1,b*1.15)
        transform=np.array([[c*sx,s*sx,12-sx*(c*cx+s*cy)],
                            [-s*sy,c*sy,16-sy*(-s*cx+c*cy)]],np.float32)
        patch=cv2.warpAffine(gray,transform,(24,32),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        chroma=np.array([colour[2]-colour[1],colour[0]-colour[1]],np.float32)
        power=float(chroma@chroma)
        if power>=64:
            colour_patch=cv2.warpAffine(image,transform,(24,32),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101).astype(np.float32)
            opponent=np.stack([colour_patch[:,:,2]-colour_patch[:,:,1],colour_patch[:,:,0]-colour_patch[:,:,1]],axis=-1)
            channel=np.clip(255*(opponent@chroma)/power,0,255)
            marks=facial_marks(channel,patch) or facial_marks(cv2.rotate(channel,cv2.ROTATE_180),cv2.rotate(patch,cv2.ROTATE_180))
            if not marks and w<=14 and h<=18:
                marks=tiny_face_bands(channel,patch) or tiny_face_bands(cv2.rotate(channel,cv2.ROTATE_180),cv2.rotate(patch,cv2.ROTATE_180))
        else:
            marks=facial_marks(patch) or facial_marks(cv2.rotate(patch,cv2.ROTATE_180))
            if not marks and w<=14 and h<=18:marks=tiny_face_bands(patch)
        if not marks:ambiguous+=1;continue
        padding=3
        if completed is not None:
            rx=np.sqrt((a*c/2)**2+(b*s/2)**2);ry=np.sqrt((a*s/2)**2+(b*c/2)**2)
            x,y,w,h=cx-rx,cy-ry,2*rx,2*ry
        candidate=clip_box(dict(x=x-padding,y=y-padding,w=w+2*padding,h=h+2*padding,
            type='face',confidence=.82,source='local_ellipse_aligned_luminance_chroma'),width,height)
        if verify_face_crop(image,candidate):boxes.append(candidate)
    boxes.sort(key=lambda b:-b['w']*b['h'])
    boxes=deduplicate(boxes,.25)
    boxes=[b for b in boxes if not any(o is not b and o['w']*o['h']>1.8*b['w']*b['h']
            and o['x']<=b['x'] and o['y']<=b['y'] and o['x']+o['w']>=b['x']+b['w']
            and o['y']+o['h']>=b['y']+b['h'] for o in boxes)]
    return boxes,ambiguous
