
from cv2 import cv2
import numpy as np

def loading_displaying_saving():
    img = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('gray', img)
    cv2.waitKey(0)
    cv2.imwrite('gray1.jpg', img)

def resizing(new_width=None, new_height=None, interp=cv2.INTER_LINEAR):
    img = cv2.imread('1.jpg')
    h, w = img.shape[:2]

    if new_width is None and new_height is None:
        return img

    if new_width is None:
        ratio = new_height / h
        dimension = (int(w * ratio), new_height)

    else:
        ratio = new_width / w
        dimension = (new_width, int(h * ratio))

    res_img = cv2.resize(img, dimension, interpolation=interp)
    cv2.imshow('pox', res_img)
    cv2.waitKey(0)


def shifting():
    img = cv2.imread('1.jpg')
    h, w = img.shape[:2]
    translation_matrix = np.float32([[1, 0, 200], [0, 1, 300]])
    dst = cv2.warpAffine(img, translation_matrix, (w, h))
    cv2.imshow('pox', dst)
    cv2.waitKey(0)

def cropping():
    img = cv2.imread('1.jpg')
    crop_img = img[10:450, 300:750]
    cv2.imshow('pox', crop_img)
    cv2.waitKey(0)

def rotation():
    img = cv2.imread('1.jpg')
    (h, w) = img.shape[:2]
    center = (int(w / 2), int(h / 2))
    rotation_matrix = cv2.getRotationMatrix2D(center, -45, 0.6)
    rotated = cv2.warpAffine(img, rotation_matrix, (w, h))
    cv2.imshow('pox', rotated)
    cv2.waitKey(0)

def cont():
    img = cv2.imread('1.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    cv2.imshow("1.jpg", img)
    cv2.waitKey(0)




loading_displaying_saving()
resizing(new_width=None, new_height=None, interp=cv2.INTER_LINEAR)
shifting()
cropping()
rotation()
cont()