import numpy as np
import cv2
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops

def glcm_features(image, distances=[1], angles=[0], levels=256):
    glcm = greycomatrix(image, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    features = {
        'contrast': greycoprops(glcm, 'contrast'),
        'homogeneity': greycoprops(glcm, 'homogeneity'),
        'energy': greycoprops(glcm, 'energy'),
        'correlation': greycoprops(glcm, 'correlation')
    }
    return features

def lbp_features(image, P=8, R=1, method='uniform'):
    lbp = local_binary_pattern(image, P, R, method)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-6)  # Normalize the histogram
    return hist

def gabor_features(image, frequencies=[0.1, 0.3, 0.5]):
    features = []
    for frequency in frequencies:
        kernel = cv2.getGaborKernel((21, 21), 8.0, 1.0, frequency, 0.5, 0, ktype=cv2.CV_32F)
        filtered_image = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        features.append(filtered_image.mean())
        features.append(filtered_image.var())
    return np.array(features)
