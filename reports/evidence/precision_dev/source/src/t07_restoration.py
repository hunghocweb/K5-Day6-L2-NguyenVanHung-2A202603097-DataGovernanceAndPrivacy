"""Gaussian PSF estimate and regularized Wiener restoration for 2D analysis."""
import cv2
import numpy as np


def estimate_sigma(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray=cv2.GaussianBlur(gray,(0,0),.6)
    dx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
    dy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    lap=cv2.Laplacian(gray,cv2.CV_32F,ksize=1)
    gradient=float(np.mean(dx*dx+dy*dy));curvature=float(np.mean(lap*lap))
    if gradient<1e-4 or curvature<1e-6:return 0.
    effective=np.sqrt(gradient/(128*curvature))
    return float(np.sqrt(max(0,effective*effective-.6*.6)))


def wiener_gaussian(image,sigma,noise_ratio=.003):
    if sigma<=0:return image.copy()
    if not np.isfinite([sigma,noise_ratio]).all() or noise_ratio<=0:raise ValueError('Finite sigma and positive noise ratio required')
    margin=max(16,int(np.ceil(6*sigma)))
    work=cv2.copyMakeBorder(image,margin,margin,margin,margin,cv2.BORDER_REFLECT_101).astype(np.float32)
    height,width=work.shape[:2]
    fy=np.fft.fftfreq(height)[:,None];fx=np.fft.rfftfreq(width)[None,:]
    transfer=np.exp(-2*np.pi*np.pi*sigma*sigma*(fx*fx+fy*fy))
    inverse=(1+noise_ratio)*transfer/(transfer*transfer+noise_ratio)
    spectrum=np.fft.rfft2(work,axes=(0,1))
    restored=np.fft.irfft2(spectrum*inverse[:,:,None],s=(height,width),axes=(0,1))
    restored=restored[margin:-margin,margin:-margin]
    return np.clip(np.rint(restored),0,255).astype(np.uint8)


def restore_for_analysis(image,gain=1.,noise_ratio=.003):
    sigma=estimate_sigma(image)
    applied=min(3.,sigma*gain) if sigma>.9 else 0.
    return wiener_gaussian(image,applied,noise_ratio),dict(estimated_sigma=sigma,applied_sigma=applied,noise_ratio=noise_ratio)
