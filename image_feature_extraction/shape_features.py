import numpy as np
import cv2
from skimage.measure import moments_hu

def fourier_descriptors(contour, num_coefficients=10):
    contour_array = contour[:, 0, :] + 1j * contour[:, 0, 1]
    fourier_result = np.fft.fft(contour_array)
    descriptors = np.abs(fourier_result)[:num_coefficients]
    return descriptors

def hu_moments(image):
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments

def zernike_moments(image, radius=21, degree=8):
    from skimage.feature import zernike_moments
    zernike_mom = zernike_moments(image, radius, degree)
    return zernike_mom
