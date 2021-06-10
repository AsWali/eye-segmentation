import torch
import numpy as np
from skimage.measure import label   
import matplotlib.pyplot as plt
import cv2

def getLargestCC(img_bw):
    labels = label(img_bw, connectivity=2, return_num=False)

    maxCC_nobcg = labels == np.argmax(np.bincount(labels.flatten(), weights=img_bw.flatten()))
    return maxCC_nobcg


def filter(image):
    image = image.numpy()
    zeros = torch.zeros(image.shape)
    for i in range(1, 4):
        x = (image <= i)
        x = x * image
        x = fillhole(x) 
        bw = getLargestCC(x)

        if i== 1:
            img_1 = (bw * i)
        elif i==2:
            img_2 = (bw * i)
        elif i==3:
            img_3 = (bw * i)
        image[zeros != 0] = 0

    zeros[img_filled_sclera > 0] = 1
    zeros[img_filled_iris > 0] = 2
    img_ret[img_filled_pupil > 0] = 3
    return zeros

def fillhole(input_image):
    '''
    input gray binary image  get the filled image by floodfill method
    Note: only holes surrounded in the connected regions will be filled.
    :param input_image:
    :return:
    '''
    im_flood_fill = input_image.copy()
    h, w = input_image[0,:,:].shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = im_flood_fill.astype("uint8")
    print(im_flood_fill.shape)
    cv2.floodFill(im_flood_fill[0,:,:], mask, (0, 0), 255)
    im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
    img_out =  np.bitwise_or(input_image, im_flood_fill_inv)
    return img_out 