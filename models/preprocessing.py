import numpy as np
import cv2


def preprocess(img, size):
    img = cv2.resize(img, size)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img
