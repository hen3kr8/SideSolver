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
    grid = find_grid(thres_image)

    corner_2(grid)
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

    # adaptive gaussian
    thres_image = cv2.adaptiveThreshold(src_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
           cv2.THRESH_BINARY_INV,53,1)
    
    if display_images_flag:
        plt.imshow(thres_image, cmap='gray')
        plt.title("thresholded image")        
        plt.show()

    return thres_image


def find_grid(image):
    #  apply a blob detecting algorithm. In this case floddfilling.

    h, w = image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    new_image = flood_filling(image)
    grid_image = find_biggest_blob(new_image)

    if display_images_flag:
        plt.imshow(grid_image, cmap='gray')
        plt.title("Extracted grid")
        plt.show()

    # cv2.imwrite( grid_image,", gray_image );
    return grid_image

def find_biggest_blob(new_image):
    # finds the longest continuous set of pixels

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
    # TODO: fix exception, find better (more efficient) way to apply floodfilling. 
    # returns all islands of pixels, where all islands have different numbers.

    h, w = image.shape[:2]
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

def search(i, j, s):
    s.append((i-1,j))
    s.append((i+1,j))
    s.append((i,j+1))
    s.append((i,j-1))



def corner_detection(image):
    # ines = cv2.HoughLines(edges,1,np.pi/180,200)
    pass
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    cv2.imwrite('houghlines3.jpg',img)



def invert(image):
    
    for i in range(0,image.shape[0]):
        for j in range(0, image.shape[1]):
            image[i,j] = int(255 if image[i,j] == 0 else 0)

    return image

def corner_2(image=None):

    #ryy harris corners
    pass
    r = cv2.HoughLines(image, 1, np.pi/180, 200)
    print(r)
    
    if display_images_flag:
        plt.imshow(image, cmap='gray')
        plt.title("Hough transform")
        plt.show()

    

if __name__ == "__main__":
    build_grid()