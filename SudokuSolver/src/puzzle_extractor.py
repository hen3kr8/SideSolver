# This program is based of a tutorial on how to make a sudoku scanner. I started this project to 
# learn more about OpenCV and Tensorflow.
# Because coding is fun. I own nothing. Don't sue me.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import operator
import sys
from skimage import data, filters
from skimage.segmentation import flood, flood_fill

display_images_flag = True
debug = True
if debug : np.set_printoptions(threshold=sys.maxsize)


def build_grid():
    raw_image = read_image()
    thres_image = apply_threshold(raw_image)
    grid = find_grid(thres_image)
    corner_detection(grid)

    pass

def read_image():

    file_name = "../sudoku_dataset-master/images/image1.jpg"
    image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE )
    
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

    # adaptive gaussian
    thres_image = cv2.adaptiveThreshold(src_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                  cv2.THRESH_BINARY_INV,53,1)
    
    if display_images_flag:
        plt.imshow(thres_image, cmap='gray')
        plt.title("thresholded image")        
        plt.show()

    return thres_image


def find_grid(image):
    #  apply a blob detecting algorithm. In this case floodfilling.

    new_image = flood_filling(image)
    grid_image = find_biggest_blob(new_image)

    if display_images_flag:
        plt.imshow(grid_image, cmap='gray')
        plt.title("Extracted grid")
        plt.show()

    return grid_image


def flood_filling(image):
    # TODO: fix exception, find better (more efficient) way to apply floodfilling. 
    # returns all islands of pixels, where all islands have different numbers.

    h, w = image.shape
    new_image = np.zeros((h,w))
    counter = 0
    s = []

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

    return new_image

def find_biggest_blob(new_image):
    # finds the longest continuous set of pixels. Each contiuous (touching) set of pixels is called
    # an island.

    h, w = new_image.shape
    unique, counts = np.unique(new_image, return_counts=True)
    z = zip(unique, counts)
    biggest_island = sorted(z, key=lambda pair: pair[1])[-2][0] #2nd last element, 1st value
    
    #convert to new_image to only contain biggest island number.
    for i in range(h):
        for j in range(w):
            new_image[i,j] = (255 if new_image[i,j] == biggest_island else 0)

    return new_image
    
def search(i, j, s):
    s.append((i-1,j))
    s.append((i+1,j))
    s.append((i,j+1))
    s.append((i,j-1))


def corner_detection(image):

    # The picture has to be in uint8 format. It was in float64.
    image_contour = image.astype('uint8') * 255

    contours, _ = cv2.findContours(image_contour, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]

    # detect index of corners in largest_contour
    bottom_right_indx, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in
                    largest_contour]), key=operator.itemgetter(1))

    top_left_indx, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in
                    largest_contour]), key=operator.itemgetter(1))

    bottom_left_indx, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in
                    largest_contour]), key=operator.itemgetter(1))

    top_right_indx, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in
                    largest_contour]), key=operator.itemgetter(1))

    bottom_right, top_left, bottom_left, top_right = [largest_contour[top_left_indx][0], 
                                                     largest_contour[top_right_indx][0], 
                                                     largest_contour[bottom_right_indx][0],
                                                     largest_contour[bottom_left_indx][0]]

    
    if display_images_flag:
        # draw corners
        image = np.asarray(image, 'uint8')
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        cv2.circle(image, tuple(bottom_right), 8, (255,0,0), -1)
        cv2.circle(image, tuple(top_left), 8, (255,0,0), -1)
        cv2.circle(image, tuple(bottom_left), 8, (255,0,0), -1)
        cv2.circle(image, tuple(top_right), 8, (255,0,0), -1)
        plt.imshow(image)
        plt.title('corners detected')
        plt.show()
    
    corners = [bottom_right, top_left, bottom_left, top_right] 
    return corners

def invert(image):
    return np.invert(image)

if __name__ == "__main__":
    build_grid()