import numpy as np
import pywt

def dwt_features(image, wavelet='haar'):
    coeffs = pywt.dwt2(image, wavelet)
    cA, (cH, cV, cD) = coeffs
    features = np.concatenate([cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()])
    return features

def wavelet_texture_features(image, wavelet='db1', level=1):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    features = []
    for coeff in coeffs:
        features.extend(coeff.flatten())
    return np.array(features)
