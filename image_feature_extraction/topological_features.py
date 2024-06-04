import numpy as np
import cv2
import gudhi as gd

def persistent_homology(image):
    skeleton = cv2.ximgproc.thinning(image)
    img_array = skeleton.astype(np.uint8)
    img_array[img_array > 0] = 1
    rips_complex = gd.RipsComplex(points=img_array)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    diag = simplex_tree.persistence()
    return diag

def betti_numbers(image):
    skeleton = cv2.ximgproc.thinning(image)
    img_array = skeleton.astype(np.uint8)
    img_array[img_array > 0] = 1
    rips_complex = gd.RipsComplex(points=img_array)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    betti_nums = simplex_tree.betti_numbers()
    return betti_nums
