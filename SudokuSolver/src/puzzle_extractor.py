# This program is based of a tutorial on how to make a sudoku scanner. I started this project to learn more about OpenCV and Tensorflow.
# Because coding is fun. I own nothing. Don't sue me.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


def build_grid():
    raw_image = read_image()
    apply_threshold(raw_image)

    pass

def read_image():

    file_name = "../sudoku_dataset-master/images/image1.jpg"
    image = cv2.imread(file_name, 2 )
    print(image)
    plt.imshow(image, cmap='gray')
    plt.show()

    return image

def blur_image(src_image):
    blurred_image = cv2.GaussianBlur(src_image, (15,15) , 1) 
    plt.imshow(blurred_image, cmap='gray')
    plt.show()
    return blurred_image

def apply_threshold(src_image):

    # src_image = blur_image(src_image)
    
    # Normal threshold
    # thres_image = cv2.threshold( src_image, 130, 255, 0 )

    now = time.time()
    # adaptive gaussian
    thres_image = cv2.adaptiveThreshold(src_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
           cv2.THRESH_BINARY,53,1)
    # adaptive mean threshold
    # thres_image = cv2.adaptiveThreshold(src_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            # cv2.THRESH_BINARY,11,2)
    

    plt.imshow(thres_image, cmap='gray')
    plt.show()
    return thres_image


if __name__ == "__main__":
    build_grid()