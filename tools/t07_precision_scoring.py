"""Exact raster rectangle metrics on compressed coordinates, for dev search.

Every floor/ceil boundary is retained; final reports still use repo metrics.
"""
import numpy as np


def _bounds(boxes,height,width):
    result=[]
    for x,y,w,h in boxes:
        left,top=max(0,int(np.floor(x))),max(0,int(np.floor(y)))
        right,bottom=min(width,int(np.ceil(x+w))),min(height,int(np.ceil(y+h)))
        if right>left and bottom>top:result.append((left,top,right,bottom))
    return result


def score_rectangles(reference,predictions):
    missed=outside=total=0
    for key,(regions,original_mask) in reference.items():
        height,width=original_mask.shape
        gt=_bounds(regions,height,width)
        red=_bounds([[b[k] for k in ['x','y','w','h']] for b in predictions[key]],height,width)
        if not gt and not red:continue
        xs=sorted({v for b in gt+red for v in [b[0],b[2]]});ys=sorted({v for b in gt+red for v in [b[1],b[3]]})
        xi={v:i for i,v in enumerate(xs)};yi={v:i for i,v in enumerate(ys)}
        areas=np.diff(ys)[:,None]*np.diff(xs)[None,:]
        rm=np.zeros(areas.shape,bool);gm=np.zeros_like(rm)
        for l,t,r,b in red:rm[yi[t]:yi[b],xi[l]:xi[r]]=True
        for l,t,r,b in gt:gm[yi[t]:yi[b],xi[l]:xi[r]]=True
        total+=int(areas[rm].sum());outside+=int(areas[rm&~gm].sum())
        for l,t,r,b in gt:
            covered=int((areas[yi[t]:yi[b],xi[l]:xi[r]]*rm[yi[t]:yi[b],xi[l]:xi[r]]).sum())
            if covered<.9*(r-l)*(b-t):missed+=1
    return dict(missed=missed,outside=outside,total=total,over=outside/total if total else 0.)
