'''
    This program is based of a tutorial on how to make a sudoku scanner.
    I started this project to learn more about OpenCV and Tensorflow.
    Because coding is fun. I own nothing. Don't sue me.
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import operator
import sys
import digit_classifier
import logging

display_images_flag = False
debug = True
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
# tf.get_logger().setLevel('INFO')

if debug:
    np.set_printoptions(threshold=sys.maxsize)
    # logging.getLogger().setLevel(logging.DEBUG)


def build_grid():
    raw_image = read_image()
    thres_image = apply_threshold(raw_image)
    grid = find_grid(thres_image)
    corners = corner_detection(grid)

    if display_images_flag:
        plot_corners_original(raw_image, corners)

    # homography
    image_homog = apply_homography(raw_image, corners)

    # extract digits
    extract_digits(image_homog)

    pass


def read_image():

    file_name = "../sudoku_dataset-master/images/image1081.jpg"
    image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

    if display_images_flag:

        plt.imshow(image, cmap='gray')
        plt.title("original image")
        plt.show()

    return image


def blur_image(src_image):

    blurred_image = cv2.GaussianBlur(src_image, (15, 15), 1)

    if display_images_flag:
        plt.imshow(blurred_image, cmap='gray')
        plt.title("blurred image")
        plt.show()

    return blurred_image


def apply_threshold(src_image):

    # src_image = blur_image(src_image)

    # adaptive gaussian
    # thres_image = cv2.adaptiveThreshold(src_image, 255,
    #               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #               cv2.THRESH_BINARY_INV,53,1)

    thres_image = cv2.adaptiveThreshold(src_image, 255,
                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                  cv2.THRESH_BINARY_INV, 11, 2)

    if display_images_flag:
        plt.imshow(thres_image, cmap='gray')
        plt.title("thresholded image")
        plt.show()

    return thres_image


def find_grid(image):
    '''
    apply a blob detecting algorithm. In this case floodfilling.
    '''

    new_image = flood_filling(image)
    grid_image = find_biggest_blob(new_image)

    if display_images_flag:
        plt.imshow(grid_image, cmap='gray')
        plt.title("Extracted grid")
        plt.show()

    return grid_image


def flood_filling(image):
    # TODO: fix exception, find better (more efficient) way to apply
    # floodfilling.
    # Returns all islands of pixels, where all islands have different numbers.

    h, w = image.shape
    new_image = np.zeros((h, w))
    counter = 0
    s = []

    for i in range(1, h-1):
        for j in range(1, w-1):

            if image[i, j] == 255 and new_image[i, j] == 0:
                counter += 1
                new_image[i, j] = counter
                search(i, j, s)

                while len(s) > 0:
                    x, y = s.pop()
                    try:
                        if image[x, y] == 255 and new_image[x, y] == 0:
                            new_image[x, y] = counter
                            search(x, y, s)

                    except IndexError:
                        pass

    return new_image


def find_biggest_blob(new_image):
    # finds the longest continuous set of pixels. Each contiuous (touching) set
    # of pixels is called an island.

    h, w = new_image.shape
    unique, counts = np.unique(new_image, return_counts=True)
    z = zip(unique, counts)
    biggest_island = sorted(z, key=lambda pair: pair[1])[-2][0]
    # 2nd last element, 1st value

    # convert to new_image to only contain biggest island number.
    for i in range(h):
        for j in range(w):
            new_image[i, j] = (255 if new_image[i, j] == biggest_island else 0)
    
    return new_image


def search(i, j, s):
    s.append((i-1, j))
    s.append((i+1, j))
    s.append((i, j+1))
    s.append((i, j-1))


def corner_detection(image):

    # The picture has to be in uint8 format. It was in float64.
    image_contour = image.astype('uint8') * 255

    contours, _ = cv2.findContours(image_contour, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
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

    top_left = largest_contour[top_left_indx][0]
    top_right = largest_contour[top_right_indx][0]
    bottom_left = largest_contour[bottom_right_indx][0]
    bottom_right = largest_contour[bottom_left_indx][0]

    if display_images_flag:
        # draw corners
        image = np.asarray(image, 'uint8')
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        cv2.circle(image, tuple(bottom_right), 8, (255, 0, 0), -1)
        cv2.circle(image, tuple(top_left), 8, (255, 0, 0), -1)
        cv2.circle(image, tuple(bottom_left), 8, (255, 0, 0), -1)
        cv2.circle(image, tuple(top_right), 8, (255, 0, 0), -1)
        plt.imshow(image)
        plt.title('corners detected')
        plt.show()

    corners = [top_left, top_right, bottom_left, bottom_right]
    return corners


def plot_corners_original(image, corners):
    # draw corners

    top_left, top_right, bottom_left, bottom_right = corners

    image = np.asarray(image, 'uint8')
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    cv2.circle(image, tuple(bottom_right), 8, (0, 0, 255), -1)
    cv2.circle(image, tuple(top_left), 8, (0, 255, 0), -1)
    cv2.circle(image, tuple(bottom_left), 8, (0, 0, 0), -1)
    cv2.circle(image, tuple(top_right), 8, (255, 0, 0), -1)
    plt.imshow(image)
    plt.title('corners detected')
    plt.show()


def apply_homography(raw_image, corners):
    # return a new image, with homography applied.

    # corners =  corrdinates on old image which will form corners of new image
    # (idea is that the grid spans the entire new image)

    corners_dst_bot_r = [raw_image.shape[1], raw_image.shape[0]]
    corners_dst_top_l = [0, 0]
    corners_dst_top_r = [raw_image.shape[1], 0]
    corners_dst_bot_l = [0, raw_image.shape[0]]

    corners_dst = [corners_dst_top_l, corners_dst_top_r, corners_dst_bot_r,
                  corners_dst_bot_l]

    # calculate homography
    h, _ = cv2.findHomography(np.array(corners), np.array(corners_dst))
    image_homog = cv2.warpPerspective(raw_image, h, (raw_image.shape[1],
                                      raw_image.shape[0]))

    if display_images_flag:
        plt.imshow(image_homog, cmap='gray')
        plt.title("applied homography")
        plt.show()

    logging.info('applied homography')
    return image_homog


def crop_center_image(image):
    '''
    We take the center 2/3 of the image of a digit and send it to the
    classifier. This way we crop out the border.
    We could have done floodfilling, but we do have a risk that the border has
    more pixels than the digit, or vice versa.
    '''

    scale_start = 1/7
    scale_end = 6/7
    im_heigth, im_width = image.shape
    start_pixel_x = int(scale_start * im_width)
    stop_pixel_x = int(scale_end * im_width)
    start_pixel_y = int(scale_start * im_heigth)
    stop_pixel_y = int(scale_end * im_heigth)

    debug = False
    cropped_image = image[start_pixel_y: stop_pixel_y, start_pixel_x: stop_pixel_x]
   
    if debug:
        print('start_pixel_x', start_pixel_x)
        print('start_pixel_y', start_pixel_y)
        print('stop_pixel_x', stop_pixel_x)
        print('stop_pixel_y', stop_pixel_y)
        print('cropped shape:', cropped_image.shape)
        print('expected shape: ', (stop_pixel_y - start_pixel_y), (stop_pixel_x - start_pixel_x)  )

    if display_images_flag:
        plt.imshow(cropped_image)
        plt.title('cropped_image')
        plt.show()

    return cropped_image


def remove_noise(image_digit):
    '''
    We apply erosion followed by dilation . This first makes the borders of the
    white pixels thinner and thickens them again. Remember to INVERT !
    This has the effect that the background noise (which are dots) is removed.

    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/
    py_imgproc/py_morphological_ops/py_morphological_ops.html
    '''

    image_digit = np.invert(image_digit)
    kernel = np.ones((5, 5), np.uint8)
    noise_free_image = cv2.morphologyEx(image_digit, cv2.MORPH_OPEN, kernel)
    # noise_free_image = cv2.erode(image_digit, kernel, iterations = 1)
    display_images_flag = True
    if display_images_flag:
        plt.imshow(noise_free_image, cmap='gray')
        plt.show()

    return noise_free_image 

def is_blank_digit(image_digit):
    '''
    Some images are blank and contain only white pixels.
    Apply flood filling, if the largest object is less than 1/10 of the total 
    image, then it is considered blank.
    '''
    image_digit = apply_threshold(image_digit)
    plt.imshow(image_digit, cmap='gray')
    plt.show()

    new_image = flood_filling(image_digit)
    unique, counts = np.unique(new_image, return_counts=True)
    z = zip(unique, counts)
    try:
        _, biggest_island_size = sorted(z, key=lambda pair: pair[1])[-2]

    except Exception as e:
    #  out of bounds, which means blank
        biggest_island_size = 0

    # 2nd last element, 1st value

    print('island size', biggest_island_size, 'np.prod(new_image.shape)', np.prod(new_image.shape))

    if biggest_island_size > np.prod(new_image.shape) * 1/10:
        print('not blank')
        return False
    
    return True


def extract_digits(image_homog):
    '''
    Divide image into 9 x 9 blocks,
    Do preprocessing before recognizing digits. (center image, erode)
    Apply biggest blob algorithm to find digit.
    '''
    
    # image_homog = apply_threshold(image_homog)
    plt.imshow(image_homog, cmap='gray')
    plt.show()

    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    image_homog = cv2.erode(image_homog, kernel)
    # image_digit = cv2.erode(image_homog, kernel)

    plt.imshow(image_homog, cmap='gray')
    plt.show()

    h, w = image_homog.shape
    image_digit_list = []

    for i in range(1, 10):
        image_digit_row_list = []

        for j in range(1, 10):
            image_digit = image_homog[int((i-1)*(h/9)):int(i*(h/9)),
                                      int((j-1)*(w/9)):int(j*(w/9))]

            # display 4th column
            if j % 10 == 8:
                # image_digit = cv2.erode(image_digit,kernel)
                plt.imshow(image_digit, cmap='gray')
                plt.show()
                noise_free_digit = crop_center_image(image_digit)
                noise_free_digit = remove_noise(noise_free_digit)
                
                if is_blank_digit(noise_free_digit):
                    noise_free_digit = np.zeros(noise_free_digit.shape) #blank
                
                else:
                    corners = []
                    # apply_homography(noise_free_digit, ) write 2nd homog method to convert small digits to a 28 x 28 plane
                    digit_classifier.predict_number(noise_free_digit)

            image_digit_row_list.append(image_digit)

        image_digit_list.append(image_digit_row_list)

    return image_digit_list

def invert(image):
    return np.invert(image)


if __name__ == "__main__":
    build_grid()
