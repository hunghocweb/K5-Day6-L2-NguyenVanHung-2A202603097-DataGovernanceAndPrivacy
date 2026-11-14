"""Evidence for jointly positive eye and mouth marks on original pixels."""
import cv2
import numpy as np
from .t07_rules_v4 import _blurred_head_models


def joint_facial_score(gray):
    vector=cv2.resize(gray.astype(np.float32),(24,32),interpolation=cv2.INTER_AREA).ravel()
    nulls=[];alternatives=[]
    for basis,eyes,mouths,cross in _blurred_head_models():
        residual=vector-basis@(basis.T@vector);error=float(residual@residual)
        er=(eyes@residual)[:,None];mr=(mouths@residual)[None,:]
        den=np.maximum(1e-5,1-cross*cross)
        ec=(er-cross*mr)/den;mc=(mr-cross*er)/den
        gain=(er*er+mr*mr-2*cross*er*mr)/den
        valid=(ec>0)&(mc>0)&(mc>ec*.10)&(mc<ec*5.)
        valid&=(ec*ec>error*.01)&(mc*mc>error*.01)
        gain=np.where(valid,gain,0)
        nulls.append(error);alternatives.append(error-float(gain.max()))
    null=min(nulls)
    if null/len(vector)<.2:return 0.
    return float(np.sqrt(min(1.,max(0.,null-min(alternatives))/max(1e-6,null))))
