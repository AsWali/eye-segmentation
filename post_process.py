import torch
import numpy as np
from skimage.measure import label   
import matplotlib.pyplot as plt
import cv2
import sys

def getLargestCC(img_bw):
    labels = label(img_bw, connectivity=2, return_num=False)

    maxCC_nobcg = labels == np.argmax(np.bincount(labels.flatten(), weights=img_bw.flatten()))
    return maxCC_nobcg


def filter(image):
    image = image.cpu().numpy() + 1
    orig_image = image
    layers = []
    for i in range(1, 4):
        bw = (image == i)
        x = fill_hole(bw[0,:,:])
        difference_with_holes = (x == bw[0,:,:])
        image = image * difference_with_holes
        layers.append(x * i)

    bw = (orig_image == 4)
    x = fill_hole(bw[0,:,:])
    difference_with_holes = (x == bw[0,:,:])
    image = image * difference_with_holes
    layers.append(x * 4)

    return np.argmax(layers, axis=0)

np.set_printoptions(threshold=sys.maxsize)

# https://www.programcreek.com/python/example/89425/cv2.floodFill
def fill_hole(input_mask):
    h, w = input_mask.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    canvas[1:h + 1, 1:w + 1] = input_mask.copy()

    mask = np.zeros((h + 4, w + 4), np.uint8)

    cv2.floodFill(canvas, mask, (0, 0), 1)
    cv2.floodFill(canvas, mask, (0, 639), 1)
    canvas = canvas[1:h + 1, 1:w + 1].astype(np.bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate((~canvas | input_mask.astype(np.uint8)) , kernel)
    eroded=cv2.erode(dilated,kernel)
    return eroded 