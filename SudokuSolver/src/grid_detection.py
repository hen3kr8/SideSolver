import logging
import operator

import config
import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_image(puzzle_image: str):

    # file_name = "../sudoku_dataset-master/images/image1024.jpg"

    image = cv2.imread(puzzle_image, cv2.IMREAD_GRAYSCALE)

    if config.display_images_flag:

        plt.imshow(image, cmap="gray")
        plt.title("original image")
        plt.show()

    return image


def blur_image(src_image: np.ndarray):

    blurred_image = cv2.GaussianBlur(src_image, (15, 15), 1)

    if config.display_images_flag:
        plt.imshow(blurred_image, cmap="gray")
        plt.title("blurred image")
        plt.show()

    return blurred_image


def apply_threshold(src_image: np.ndarray, bin: bool = False):

    thres_image = None
    if bin:
        blur = cv2.GaussianBlur(src_image, (3, 3), 0)
        ret, thres_image = cv2.threshold(
            blur, thresh=0, maxval=127, type=cv2.THRESH_BINARY_INV
        )
        thres_image = np.invert(thres_image)

    else:
        thres_image = cv2.adaptiveThreshold(
            src_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )
    if config.display_images_flag:
        plt.imshow(thres_image, cmap="gray")
        plt.title("thresholded image 11")
        plt.show()

    return thres_image


def find_largest_object(image: np.ndarray):
    """apply a blob detecting algorithm. In this case floodfilling.

    """

    new_image = flood_filling(image)
    grid_image = find_biggest_blob(new_image)

    if config.display_images_flag:
        plt.imshow(grid_image, cmap="gray")
        plt.title("Extracted grid")
        plt.show()

    return grid_image


def flood_filling(image: np.ndarray):
    # TODO(fix): fix exception, find better (more efficient) way to apply
    # floodfilling.
    # Returns all islands of pixels, where all islands have different numbers.

    h, w = image.shape
    new_image = np.zeros((h, w))
    counter = 0
    s = []

    for i in range(1, h - 1):
        for j in range(1, w - 1):

            if image[i, j] == 255 and new_image[i, j] == 0:
                counter += 1
                new_image[i, j] = counter
                search(i, j, s, h, w)

                while len(s) > 0:
                    x, y = s.pop()
                    try:
                        if image[x, y] == 255 and new_image[x, y] == 0:
                            new_image[x, y] = counter
                            search(x, y, s, h, w)

                    except IndexError:
                        pass

    return new_image


def find_biggest_blob(new_image: np.ndarray, largest_island: int = 2):

    """Finds the longest continuous (touching) set of pixels.

    Arguments:
        new_image {np.array} -- image in which each continuous set of pixels
                                is a unique number (a set of continuous pixels
                                can be thought of as a unique island).

    Keyword Arguments:
        largest_island {int} -- originally the function was only used to find
                                the grid, which was the 2nd largest shape on
                                the image, after the background.
                                (default: {2})

    Returns:
        [np.array] -- the image only containing the biggest blob.
    """
    unique, counts = np.unique(new_image, return_counts=True)
    if len(unique) < largest_island:
        logging.debug("no blob found - cell is blank")
        return np.zeros_like(new_image)

    biggest_island = unique[np.argsort(counts, kind="stable")[-largest_island]]
    return np.where(new_image == biggest_island, 255, 0).astype(new_image.dtype)

    # h, w = new_image.shape
    # unique, counts = np.unique(new_image, return_counts=True)
    # z = zip(unique, counts)
    # try:
    #     biggest_island = sorted(z, key=lambda pair: pair[1])[-largest_island][0]
    #     # 2nd last element, 1st value
    #     # convert to new_image to only contain biggest island number.
    #     for i in range(h):
    #         for j in range(w):
    #             new_image[i, j] = 255 if new_image[i, j] == biggest_island else 0

    # except IndexError as e:
    #     logging.error("%s - Could not find digit.", e)


    # return new_image


def search(i: int, j: int, s: list[tuple[int, int]], h: int, w: int):
    if i > 0:     s.append((i - 1, j))
    if i < h - 1: s.append((i + 1, j))
    if j < w - 1: s.append((i, j + 1))
    if j > 0:     s.append((i, j - 1))


def corner_detection(image: np.ndarray):

    # The picture has to be in uint8 format. It was in float64.
    image_contour = image.astype("uint8") * 255

    # ret, image_contour = cv2.threshold(image_contour, 127, 255, 0)


    contours, _ = cv2.findContours(
        image_contour, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        logging.warning("no contours found, cannot detect grid corners")
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]

    # detect index of corners in largest_contour
    bottom_right_indx, _ = max(
        enumerate([pt[0][0] + pt[0][1] for pt in largest_contour]),
        key=operator.itemgetter(1),
    )

    top_left_indx, _ = min(
        enumerate([pt[0][0] + pt[0][1] for pt in largest_contour]),
        key=operator.itemgetter(1),
    )

    bottom_left_indx, _ = min(
        enumerate([pt[0][0] - pt[0][1] for pt in largest_contour]),
        key=operator.itemgetter(1),
    )

    top_right_indx, _ = max(
        enumerate([pt[0][0] - pt[0][1] for pt in largest_contour]),
        key=operator.itemgetter(1),
    )

    top_left = largest_contour[top_left_indx][0]
    top_right = largest_contour[top_right_indx][0]
    bottom_left = largest_contour[bottom_right_indx][0]
    bottom_right = largest_contour[bottom_left_indx][0]

    if config.display_images_flag:
        # draw corners
        image = np.asarray(image, "uint8")
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        cv2.circle(image, tuple(bottom_right), 8, (255, 0, 0), -1)
        cv2.circle(image, tuple(top_left), 8, (255, 0, 0), -1)
        cv2.circle(image, tuple(bottom_left), 8, (255, 0, 0), -1)
        cv2.circle(image, tuple(top_right), 8, (255, 0, 0), -1)
        plt.imshow(image)
        plt.title("corners detected")
        plt.show()

    corners = [top_left, top_right, bottom_left, bottom_right]

    quad_area = cv2.contourArea(np.array(corners, dtype=np.int32))
    blob_area = cv2.contourArea(largest_contour)

    if blob_area == 0:
        logging.warning("largest contour has zero area, cannot detect grid corners")
        return None

    ratio = quad_area / blob_area

    if ratio > 1.4: # some corners are not detected correctly, so we skip this image. The ratio of the area of the quad to the area of the blob should be less than 1.4.
        logging.warning(
            "The ratio of the area of the quad to the area of the blob is %s, which is greater than 1.4. This might indicate that the corners are not detected correctly.",
            ratio,
        )
        return None

    return corners


def plot_corners_original(image: np.ndarray, corners: list):
    # draw corners

    top_left, top_right, bottom_left, bottom_right = corners

    image = np.asarray(image, "uint8")
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    cv2.circle(image, tuple(bottom_right), 8, (0, 0, 255), -1)
    cv2.circle(image, tuple(top_left), 8, (0, 255, 0), -1)
    cv2.circle(image, tuple(bottom_left), 8, (0, 0, 0), -1)
    cv2.circle(image, tuple(top_right), 8, (255, 0, 0), -1)
    plt.imshow(image)
    plt.title("corners detected")
    plt.show()


def apply_homography(raw_image: np.ndarray, corners_src: list | None = None, corners_dst: list | None = None, new_image: np.ndarray | None = None):
    """Return a new image, with homography applied.

    Arguments:
        raw_image {np.array} -- raw image to which homography is applied.
        corners_src {list of tuples} -- coordinates on old image
        which will form corners of new image
        corners_dst {tuple}  -- corners of new image
        new_image {np.array} -- new array to be returned, needed for shape

    Returns:
        [np.array] -- resulting new image to which homography was applied.
    """

    # (idea is that the grid spans the entire new image, so technically shape
    # stays the same
    corners_top_l = [0, 0]
    corners_top_r = [raw_image.shape[1], 0]
    corners_bot_r = [raw_image.shape[1], raw_image.shape[0]]
    corners_bot_l = [0, raw_image.shape[0]]
    corners = [corners_top_l, corners_top_r, corners_bot_r, corners_bot_l]

    if corners_dst is None:
        # this is the case when the grid is extracted from the image.
        corners_dst = corners
        h, _ = cv2.findHomography(np.array(corners_src), np.array(corners_dst))
        image_homog = cv2.warpPerspective(
            raw_image, h, (raw_image.shape[1], raw_image.shape[0])
        )

    if corners_src is None:
        # this is the case when digits are to be reshaped to 28 x 28 images.
        corners_src = corners
        h, _ = cv2.findHomography(np.array(corners_src), np.array(corners_dst))
        image_homog = cv2.warpPerspective(
            raw_image, h, (new_image.shape[1], new_image.shape[0])
        )

    if config.display_images_flag:
        plt.imshow(image_homog, cmap="gray")
        plt.title("applied homography")
        plt.show()

    logging.debug("applied homography")
    return image_homog
