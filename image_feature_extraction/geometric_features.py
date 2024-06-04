import numpy as np
import cv2

def curvature_features(contour):
    curvature = []
    for i in range(len(contour)):
        p1 = contour[i - 1]
        p2 = contour[i]
        p3 = contour[(i + 1) % len(contour)]
        angle = np.arctan2(p3[0, 1] - p2[0, 1], p3[0, 0] - p2[0, 0]) - np.arctan2(p1[0, 1] - p2[0, 1], p1[0, 0] - p2[0, 0])
        curvature.append(angle)
    return np.array(curvature)

def fractal_dimension(image, threshold=0.9):
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where(S > 0)[0])

    Z = (image < threshold)
    p = min(Z.shape)
    n = 2**np.floor(np.log(p)/np.log(2))
    sizes = 2**np.arange(np.log(n)/np.log(2), 1, -1)
    counts = [boxcount(Z, size) for size in sizes]
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]
