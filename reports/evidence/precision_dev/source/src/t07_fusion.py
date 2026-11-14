"""Fuse independent mathematical proposals without any image/label lookup."""
from .t07_detectors import clip_box, overlap
from .t07_router import _coverage_preserving_nms


def fuse(coarse,fine,small,width,height,face_fraction=0.,plate_x_fraction=.15):
    faces=[]
    for b in coarse:
        if b['type']!='face':continue
        if max(b['w'],b['h'])<=30:
            faces.append(clip_box(dict(b,x=b['x']-1,y=b['y']-1,w=b['w']+2,h=b['h']+2),width,height))
        else:faces.append(dict(b))
    faces += [dict(b) for b in small if b['type']=='face']
    faces += [clip_box(dict(b),width,height,face_fraction) for b in fine if b['type']=='face']
    faces=_coverage_preserving_nms(faces,width,height,.25)
    plates=[]
    for candidate in [b for b in fine if b['type']=='plate']:
        supported=[b for b in coarse if b['type']=='plate' and overlap(b,candidate)>=.15]
        if not supported:
            plates.append(dict(candidate));continue
        expanded=clip_box(dict(candidate,x=candidate['x']-candidate['w']*plate_x_fraction,
                                w=candidate['w']*(1+2*plate_x_fraction)),width,height)
        additions=[dict(candidate)]
        for other in supported:
            x,y=max(expanded['x'],other['x']),max(expanded['y'],other['y'])
            right=min(expanded['x']+expanded['w'],other['x']+other['w'])
            bottom=min(expanded['y']+expanded['h'],other['y']+other['h'])
            if right>x and bottom>y:additions.append(dict(candidate,x=x,y=y,w=right-x,h=bottom-y,source='supported_plate_text_extent'))
        plates.extend(_coverage_preserving_nms(additions,width,height,.1))
    return faces+_coverage_preserving_nms(plates,width,height,.3)
