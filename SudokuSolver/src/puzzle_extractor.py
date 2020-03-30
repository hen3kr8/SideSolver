# This program is based of a tutorial on how to make a sudoku scanner. I started this project to 
# learn more about OpenCV and Tensorflow.
# Because coding is fun. I own nothing. Don't sue me.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage import data, filters
from skimage.segmentation import flood, flood_fill

display_images_flag = True


def build_grid():
    raw_image = read_image()
    thres_image = apply_threshold(raw_image)
    find_grid(thres_image)

    pass

def read_image():

    file_name = "../sudoku_dataset-master/images/image1.jpg"
    image = cv2.imread(file_name, 2 )
    
    if display_images_flag:

        plt.imshow(image, cmap='gray')
        plt.title("original image")
        plt.show()

    return image

def blur_image(src_image):
    blurred_image = cv2.GaussianBlur(src_image, (15,15) , 1) 

    if display_images_flag:
        plt.imshow(blurred_image, cmap='gray')
        plt.title("blurred image")
        plt.show()

    return blurred_image

def apply_threshold(src_image):

    # src_image = blur_image(src_image)
    # Normal threshold
    # thres_image = cv2.threshold( src_image, 130, 255, 0 )

    # adaptive gaussian
    thres_image = cv2.adaptiveThreshold(src_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
           cv2.THRESH_BINARY_INV,53,1)
    
    # adaptive mean threshold
    # thres_image = cv2.adaptiveThreshold(src_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            # cv2.THRESH_BINARY,11,2)
    
        
    if display_images_flag:
        plt.imshow(thres_image, cmap='gray')
        plt.title("thresholded image")        
        plt.show()

    return thres_image



def find_grid(image):
    #  apply a blob detecting algorithm. In this case floddfilling.

    h, w = image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # flood_fill_image = cv2.floodFill(image, mask, None, (255, 255, 255), 0*3, 0*3, 4)

    # # flood_fill_image = cv2.floodFill(image, (3,3), (0,0))
    # plt.imshow(flood_fill_image, cmap='gray')
    # plt.title("floodfill image")
    # plt.show()    
    new_image = flood_filling(image)
    grid_image = find_biggest_blob(new_image)

    plt.imshow(grid_image, cmap='gray')
    plt.title("Extracted grid")
    plt.show()

    # final_image = flood_fill(image, new_image, 48, 15, 255, 255)
    # print(new_image)
    
    # if display_images_flag:
        # plt.imshow(flood_fill_image, cmap='gray')

def find_biggest_blob(new_image):
    #finds the longest continuous set of pixels

    h,w = new_image.shape
    unique, counts = np.unique(new_image, return_counts=True)
    z = zip(unique, counts)
    biggest_island = sorted(z, key=lambda pair: pair[1])[-2][0] #2nd last element, 1st value
    #convert to new_image to only contain biggest island number.
    for i in range(h):
        for j in range(w):
            new_image[i,j] = (255 if new_image[i,j] == biggest_island else 0)

    return new_image

def flood_filling(image):

    # returns all islands of pixels, where all islands have different numbers.

    h, w = image.shape[:2]
    new_image = np.zeros((h,w))
    counter = 0
    s = []

    # boarder not included, strictly speaking it should, but if thicker than 1 pixel, it will get picked up
    for i in range(1, h-1):
        for j in range(1,w-1):
            
            if image[i,j] == 255 and new_image[i,j] == 0:
                counter += 1
                new_image[i,j] = counter 
                
                search(i, j, s)
                while len(s) > 0:
                    x,y = s.pop()
                    try: 
                        if image[x,y] == 255 and new_image[x,y] == 0:
                            new_image[x,y] = counter
                            search(x, y, s)

                    except IndexError :
                        pass
                        # print('border : ', x, y, e)

    return new_image

def search(i, j, s):
    s.append((i-1,j))
    s.append((i+1,j))
    s.append((i,j+1))
    s.append((i,j-1))
    

if __name__ == "__main__":
    build_grid()