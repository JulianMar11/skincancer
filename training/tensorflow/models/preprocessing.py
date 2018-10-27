import numpy as np
import cv2


def preprocess(img, size):
    img = augment(img)
    img = cv2.resize(img, size)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img


def augment(img):
    img = rotate(img)
    img = flip(img)
    img = change_values(img)
    return img


def rotate(img):
    degree = 0
    if np.random.random_integers(0,1) == 1:
        degree += 90
    if np.random.random_integers(0, 1) == 1:
        degree += 90
    if np.random.random_integers(0, 1) == 1:
        degree += 90
    rows,cols,colors = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),degree,1)
    img = cv2.warpAffine(img, M, (cols, rows))
    return img


def flip(img):
    a = np.random.random_integers(0,1)
    b = np.random.random_integers(0,1)
    if a == 1:
        img=cv2.flip(img,1)
    if b == 1:
        img=cv2.flip(img,0)
    return img


def change_values(img):
    huefactor = hue()
    valuefactor = value()
    saturationfactor = saturation()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  #cv2.COLOR_BGR2HSV
    width, height, channels = img.shape
    for x in range(0,width):
        for y in range(0,height):
            img[x,y,0] = max(min(img[x,y,0] + huefactor,179),0)    #hue
            img[x,y,1] = max(min(img[x,y,1] + saturationfactor,255),0)   #saturation
            img[x,y,2] = max(min(img[x,y,2] + valuefactor,255),0)     #value
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)  #cv2.COLOR_BGR2HSV
    return img


def hue():
    scalefactor = np.random.normal(0,4,1)[0]
    scalefactor = min(10, scalefactor)
    scalefactor = max(-10, scalefactor)
    return scalefactor


def saturation():
    scalefactor = np.random.normal(0,8,1)[0]
    scalefactor = min(20, scalefactor)
    scalefactor = max(-20, scalefactor)
    return scalefactor


def value():
    scalefactor = np.random.normal(0,8,1)[0]
    scalefactor = min(20, scalefactor)
    scalefactor = max(-20, scalefactor)
    return scalefactor
