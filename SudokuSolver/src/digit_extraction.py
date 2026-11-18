import logging

import config
import cv2
import grid_detection
import matplotlib.pyplot as plt
import numpy as np


def extract_digits(image_homog: np.ndarray):
    """Divide image into 9 x 9 blocks,

    We do preprocessing before recognizing digits. (center image, erode)
    Apply biggest blob algorithm to find digit.

    Arguments:
        image_homog {np.array} -- image

    Returns:
        [list of lists of np.array] -- a matrix of images of digits
    """

    # plt.imshow(image_homog, cmap="gray")
    # plt.show()

    h, w = image_homog.shape
    image_digit_list = []

    for i in range(1, 10):
        image_digit_row_list = []

        for j in range(1, 10):
            image_digit = image_homog[
                int((i - 1) * (h / 9)) : int(i * (h / 9)),
                int((j - 1) * (w / 9)) : int(j * (w / 9)),
            ]

            image_digit_row_list.append(image_digit)

        image_digit_list.append(image_digit_row_list)

    return image_digit_list


def crop_center_image(image: np.ndarray):
    """Crop out the border by taking the center 5/7 of the image.

    We could have done floodfilling, but we do have a risk that the border has
    more pixels than the digit, or vice versa.

    Arguments:
        image {np.array} -- image of digit

    Returns:
        [np.array] -- cropped image
    """

    scale_start = 1 / 7
    scale_end = 6 / 7
    im_heigth, im_width = image.shape
    start_pixel_x = int(scale_start * im_width)
    stop_pixel_x = int(scale_end * im_width)
    start_pixel_y = int(scale_start * im_heigth)
    stop_pixel_y = int(scale_end * im_heigth)

    cropped_image = image[start_pixel_y:stop_pixel_y, start_pixel_x:stop_pixel_x]

    if config.debug:
        print("start_pixel_x", start_pixel_x)
        print("start_pixel_y", start_pixel_y)
        print("stop_pixel_x", stop_pixel_x)
        print("stop_pixel_y", stop_pixel_y)
        print("cropped shape:", cropped_image.shape)
        print(
            "expected shape: ",
            (stop_pixel_y - start_pixel_y),
            (stop_pixel_x - start_pixel_x),
        )

    if config.display_images_flag:
        plt.imshow(cropped_image)
        plt.title("cropped_image")
        plt.show()

    return cropped_image


def remove_noise(image_digit: np.ndarray):
    """We apply erosion followed by dilation.

    This has the effect that the background noise (which are dots) is removed.
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/
    py_imgproc/py_morphological_ops/py_morphological_ops.html

    Remember to INVERT !

    Arguments:
        image_digit {np.array} -- image of digit

    Returns:
        [np.array] -- image of digit with noise removed.
    """

    image_digit = np.invert(image_digit)
    kernel = np.ones((3, 3), np.uint8)
    noise_free_image = cv2.morphologyEx(image_digit, cv2.MORPH_OPEN, kernel)
    # noise_free_image = cv2.erode(image_digit, kernel, iterations=2)

    if config.display_images_flag:
        plt.imshow(noise_free_image, cmap="gray")
        plt.title("removed all noise before prediction")
        plt.show()

    noise_free_image = np.invert(noise_free_image)
    return noise_free_image


def is_blank_digit(image_digit: np.ndarray):
    """Determine whether image is a digit or blank.

    Apply flood filling, if the largest object is less than 1/10 of the
    total image, then it is considered blank.

    Arguments:
        image_digit {np.array} -- image of digit

    Returns:
        [Boolean]
    """

    new_image = grid_detection.flood_filling(image_digit)
    unique, counts = np.unique(new_image, return_counts=True)
    z = zip(unique, counts)
    try:
        _, biggest_island_size = sorted(z, key=lambda pair: pair[1])[-2]

    except IndexError:
        # out of bounds, which means blank
        biggest_island_size = 0

    logging.debug(
        "island size: %s np.prod(new_image.shape) %s",
        biggest_island_size,
        np.prod(new_image.shape),
    )

    if biggest_island_size > np.prod(new_image.shape) * 1 / 40:
        logging.debug("not blank")
        return False

    else:
        logging.debug("blank")
        return True


def reshape_digit_image(image: np.ndarray, new_image_shape: tuple[int, int] = (32, 32), margin: int = 4):
    """Reshape image to 32 x 32 image using homography. Strike that.
    Homography  preserves straght lines, not the scale/ width height ratio. So we will scale using open cv resze

    This classifier was trained with MNIST which is 32 x 32.

    Arguments:
        image {np.array} -- image of digit to be reshaped

    Keyword Arguments:
        new_image_shape {tuple} -- In future, we might train classifier with
                                   different dataset containing images of
                                   different shape (default: {(32, 32)})
        margin {int} -- Margin to add around the digit (default: {4})

    Returns:
        [np.array] -- reshaped image
    """

    h, w = image.shape
    side = new_image_shape[0] - 2 * margin
    scale = side / max(h, w)                      # ONE factor, both axes
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)

    reshaped_image = np.zeros(new_image_shape)
    top, left = (new_image_shape[0] - nh) // 2, (new_image_shape[1] - nw) // 2
    reshaped_image[top:top + nh, left:left + nw] = resized


    # show image
    display_images_flag = False
    if display_images_flag:
        plt.imshow(reshaped_image, cmap="gray")
        plt.title("reshaped_image")
        plt.show()

    return reshaped_image




def process_digit_images_before_classification(image_digit_list: list[list[np.ndarray]]):
    """Remove grid border, reshape/crop digit, remove noise via floodfilling.

    0. Crop border
    1. Apply floodfilling to find digit - this removes all noise
    2. Center image using result from 1.
    3. Reshape using scale

    Arguments:
        image_digit_list {list of list of np.array} -- 9 x 9 list of lists, each
                                                       containing an image of a digit

    Return:
        {list of list of np.array} -- processed, de-noised images
    """

    display_images_flag = False
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], np.uint8)
    processed_images_digit = np.zeros((9, 9), dtype=object)

    for i in range(0, 9):
        for j in range(0, 9):
            image_digit = image_digit_list[i][j]
            # re = np.zeros((32,32))
            reshaped_image = 0

            # if j % 10 == 7 and i == 0:
            if display_images_flag:
                plt.imshow(image_digit)
                plt.title("image_digit")
                plt.show()

            cropped_image = crop_center_image(image_digit)
            thres_image = grid_detection.apply_threshold(cropped_image)
            digit = grid_detection.find_largest_object(thres_image)

            flood_filled_image = cv2.morphologyEx(digit, cv2.MORPH_CLOSE, kernel)

            if display_images_flag:
                plt.imshow(flood_filled_image)
                plt.title("flood_filled_image")
                plt.show()

            if not is_blank_digit(flood_filled_image):
                centered_digit = center_image(flood_filled_image, cropped_image)
                # centered_digit = cv2.erode(centered_digit, kernel)
                reshaped_image = reshape_digit_image(centered_digit)

            processed_images_digit[i][j] = reshaped_image

    logging.info("done processing all digits")
    return processed_images_digit


def center_image(flood_filled_image: np.ndarray, cropped_image: np.ndarray):
    # find center in floodfilled
    display_images_flag = False
    h, w = flood_filled_image.shape

    # the below code shows which rows and columns are completely blank and can
    # be removed
    rows_containing_digit = np.sum(flood_filled_image, axis=1)
    cols_containing_digit = np.sum(flood_filled_image, axis=0)

    new_image = np.zeros(shape=(flood_filled_image.shape))

    new_image = flood_filled_image.copy()


    if display_images_flag:
        plt.imshow(new_image)
        plt.title("new_image")
        plt.show()

    # remove blank lines
    rows_to_remove = [i for (i, v) in enumerate(rows_containing_digit) if v == 0]
    cols_to_remove = [i for (i, v) in enumerate(cols_containing_digit) if v == 0]

    centered_image = np.delete(new_image, rows_to_remove, axis=0)
    centered_image = np.delete(centered_image, cols_to_remove, axis=1)

    logging.debug("Empty lines removed")
    if display_images_flag:
        plt.imshow(centered_image, cmap="gray")
        plt.title("centered_image")
        plt.show()


    logging.debug("padding added")
    return centered_image
