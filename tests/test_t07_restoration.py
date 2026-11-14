import cv2
import numpy as np

from src.t07_restoration import estimate_sigma,wiener_gaussian


def test_wiener_preserves_shape_dtype_and_constant_dc():
    image=np.full((80,120,3),(30,80,150),np.uint8)
    restored=wiener_gaussian(image,1.5)
    np.testing.assert_array_equal(restored,image)


def test_regularized_inverse_improves_known_blurred_geometry():
    image=np.full((120,180,3),70,np.uint8)
    cv2.rectangle(image,(30,35),(130,90),(220,220,220),-1)
    cv2.putText(image,'AB12',(45,70),cv2.FONT_HERSHEY_SIMPLEX,.8,(15,15,15),2)
    blurred=cv2.GaussianBlur(image,(0,0),1.8)
    restored=wiener_gaussian(blurred,1.8)
    before=float(np.mean((blurred.astype(float)-image)**2))
    after=float(np.mean((restored.astype(float)-image)**2))
    assert after<before
    assert estimate_sigma(blurred)>estimate_sigma(image)
