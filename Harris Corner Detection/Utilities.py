import cv2
import numpy as np

def get_gradients_xy(img, ksize):
    #CV_16S will keep the sign of the edges
    sobelx = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=ksize)
    sobelx = np.absolute(sobelx)
    sobely = np.absolute(sobely)
    sobelx = np.uint8(sobelx)
    sobely = np.uint8(sobely)
    return sobelx, sobely



def rescale(img, min,max):
    ## Student Code ~ 2 lines of code for img normalization

    for i in range(img.shape[1]):
        img[:, i] = ((img[:, i] - np.min(img[:, i])) / (np.max(img[:, i]) - np.min(img[:, i]))) * (max - min) + min



    ## End of Student Code
    return img
