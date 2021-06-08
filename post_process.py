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
    zeros = np.zeros(image.shape)
    for i in range(0, 4):
        x = (image <= i)
        x = x * image     
        imgplot = plt.imshow(x[0,:,:])
        plt.show()
        x = img_fill(x, i)      
        imgplot = plt.imshow(x[0,:,:])
        plt.show()
        bw = getLargestCC(x)

        # if i== 1:
        #     img_1 = (bw * i)
        #     image[img_1 != 0] = 0  
        # elif i==2:
        #     img_2 = (bw * i)
        #     image[img_2 != 0] = 0  
        # elif i==3:
        #     img_3 = (bw * i) 
        #     image[img_3 != 0] = 0  

    zeros[img_1 > 0] = 1
    zeros[img_2 > 0] = 2
    zeros[img_3 > 0] = 3
    return zeros

def img_fill(im_in,n):   # n = binary image threshold
    th, im_th = cv2.threshold(im_in, n, 255, cv2.THRESH_BINARY);
     
    # Copy the thresholded image.
    im_floodfill = im_th.copy()
     
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
     
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
     
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
     
    # Combine the two images to get the foreground.
    fill_image = im_th | im_floodfill_inv
    
    return fill_image 