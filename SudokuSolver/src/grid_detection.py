import logging
import operator
import statistics

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


def detect_grid_corners(raw_image: np.ndarray):
    """Find the sudoku grid's 4 corners, preferring the original floodfill +
    contour approach (corner_detection()) and falling back to a Hough-line-
    based approach when that fails.

    The floodfill approach requires the grid lines to form one single
    connected blob of pixels, which is fragile: a few small breaks in faint
    or noisy grid lines (common in real photos) splits it into disconnected
    fragments instead, and the "biggest fragment" is then not the actual
    grid boundary. The Hough approach doesn't have this requirement -- it
    accumulates votes across colinear edge pixels directly, so disconnected
    line fragments still vote for the same line. It's used only as a
    fallback (not the primary method) because, unlike the floodfill/contour
    approach's single confidence check, it has no equally strong way to
    detect when it's found the wrong quad, only when it's found nothing at
    all.

    Arguments:
        raw_image {np.ndarray} -- grayscale raw puzzle photo

    Returns:
        [list] -- [top_left, top_right, bottom_left, bottom_right], or None
                   if neither approach found a confident grid boundary.
    """

    thres_image = apply_threshold(raw_image)
    grid = find_largest_object(thres_image)
    corners = corner_detection(grid)

    if corners is not None:
        return corners

    logging.info("floodfill grid detection failed, falling back to Hough line detection")
    return detect_grid_corners_hough(raw_image)


def detect_grid_corners_hough(raw_image: np.ndarray):
    """Find the sudoku grid's 4 corners using a Hough line transform.

    Finds strong straight lines in the image, keeps the ones that are close
    to horizontal or vertical, and looks for the longest run of lines whose
    spacing is consistent with a regular 9x9 grid (tolerating the occasional
    missed internal line) to tell grid lines apart from unrelated straight
    lines elsewhere in the photo (page edges, header text, etc). The first
    and last line of that run, in each orientation, are taken as the grid's
    boundary.

    Arguments:
        raw_image {np.ndarray} -- grayscale raw puzzle photo

    Returns:
        [list] -- [top_left, top_right, bottom_left, bottom_right], or None
                   if no confident grid boundary was found.
    """

    h, w = raw_image.shape
    min_dim = min(h, w)

    blurred = cv2.GaussianBlur(raw_image, (5, 5), 0)
    canny_low, canny_high = _hough_canny_thresholds(blurred)
    edges = cv2.Canny(blurred, canny_low, canny_high)

    vote_threshold = max(30, int(0.17 * min_dim))
    lines = cv2.HoughLines(edges, 1, np.pi / 180, vote_threshold)

    if lines is None:
        logging.warning("Hough transform found no lines, cannot detect grid corners")
        return None

    horizontal, vertical = _split_hough_lines_by_orientation(lines)

    if len(horizontal) < 3 or len(vertical) < 3:
        logging.warning(
            "Hough transform found too few grid-aligned lines (%d horizontal, "
            "%d vertical), cannot detect grid corners",
            len(horizontal), len(vertical),
        )
        return None

    horizontal_positions = [(line, _hough_line_position(line, w, h, vertical=False)) for line in horizontal]
    vertical_positions = [(line, _hough_line_position(line, w, h, vertical=True)) for line in vertical]

    merge_dist = 0.03 * min_dim
    horizontal_lines = _merge_nearby_hough_lines(horizontal_positions, merge_dist)
    vertical_lines = _merge_nearby_hough_lines(vertical_positions, merge_dist)

    horizontal_run = _longest_regular_hough_run(horizontal_lines)
    vertical_run = _longest_regular_hough_run(vertical_lines)

    if len(horizontal_run) < 3 or len(vertical_run) < 3:
        logging.warning(
            "Hough transform could not find a regularly-spaced run of grid "
            "lines, cannot detect grid corners"
        )
        return None

    top, bottom = horizontal_run[0][0], horizontal_run[-1][0]
    left, right = vertical_run[0][0], vertical_run[-1][0]

    corners = [
        _hough_line_intersection(top, left),
        _hough_line_intersection(top, right),
        _hough_line_intersection(bottom, left),
        _hough_line_intersection(bottom, right),
    ]

    if any(corner is None for corner in corners):
        logging.warning("Hough grid lines did not intersect, cannot detect grid corners")
        return None

    quad_area = cv2.contourArea(np.array(corners, dtype=np.float32)[[0, 1, 3, 2]])
    if quad_area / (h * w) < 0.1:
        logging.warning(
            "Hough-detected grid quad is implausibly small relative to the "
            "image, cannot detect grid corners"
        )
        return None

    if config.display_images_flag:
        image = cv2.cvtColor(raw_image.astype("uint8"), cv2.COLOR_GRAY2RGB)
        pts = np.array(corners, dtype=np.int32)
        for point in pts:
            cv2.circle(image, tuple(point), 8, (255, 0, 0), -1)
        cv2.polylines(image, [pts[[0, 1, 3, 2]]], True, (0, 255, 0), 3)
        plt.imshow(image)
        plt.title("corners detected (Hough)")
        plt.show()

    return [np.array(corner, dtype=np.int32) for corner in corners]


def _hough_canny_thresholds(image: np.ndarray, percentile: float = 92, ratio: float = 0.4):
    """Pick Canny edge thresholds from the image's own gradient magnitude
    distribution rather than fixed values or raw pixel intensity. Raw pixel
    intensity doesn't work here: puzzle photos vary a lot in contrast (some
    have a pixel range as narrow as ~50-180 rather than 0-255), and a fixed
    or intensity-based threshold tuned for one ends up finding zero edges in
    the other.
    """
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    upper = np.percentile(magnitude, percentile)
    lower = ratio * upper
    return int(lower), int(upper)


def _split_hough_lines_by_orientation(lines: np.ndarray, angle_tolerance_deg: float = 15):
    """Split cv2.HoughLines() output into near-horizontal and near-vertical
    lines, discarding anything at a diagonal (not part of an upright grid).
    """
    horizontal, vertical = [], []
    tolerance = np.deg2rad(angle_tolerance_deg)

    for line in lines:
        rho, theta = line[0]
        if abs(theta - np.pi / 2) <= tolerance:
            horizontal.append((rho, theta))
        elif theta <= tolerance or theta >= np.pi - tolerance:
            vertical.append((rho, theta))

    return horizontal, vertical


def _hough_line_position(line: tuple, image_width: int, image_height: int, vertical: bool) -> float:
    """Approximate where a Hough line (rho, theta) crosses the image's
    vertical (for a vertical line) or horizontal (for a horizontal line)
    midpoint, as a single comparable position regardless of theta's sign
    convention.
    """
    rho, theta = line
    if vertical:
        return (rho - (image_height / 2) * np.sin(theta)) / np.cos(theta)
    return (rho - (image_width / 2) * np.cos(theta)) / np.sin(theta)


def _hough_line_intersection(line1: tuple, line2: tuple):
    """Intersection point of two Hough lines (rho, theta), or None if they're
    (near enough) parallel.
    """
    rho1, theta1 = line1
    rho2, theta2 = line2

    a = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)],
    ])
    b = np.array([rho1, rho2])

    try:
        x, y = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return None

    return (x, y)


def _merge_nearby_hough_lines(lines_with_position: list, merge_dist: float) -> list:
    """Hough regularly detects several near-duplicate lines for the same
    real line. Collapse any lines within merge_dist of each other into one,
    keeping one representative line and the average position.
    """
    sorted_lines = sorted(lines_with_position, key=lambda item: item[1])

    clusters = []
    for line, position in sorted_lines:
        if clusters and position - clusters[-1][-1][1] <= merge_dist:
            clusters[-1].append((line, position))
        else:
            clusters.append([(line, position)])

    return [
        (cluster[0][0], sum(position for _, position in cluster) / len(cluster))
        for cluster in clusters
    ]


def _longest_regular_hough_run(lines_with_position: list, tolerance: float = 0.3) -> list:
    """Find the longest run of lines whose gaps are each close to a small
    integer multiple of a common base unit (the grid cell size). This
    tolerates the occasional missed internal line (gap = 2x or 3x the base
    unit) while rejecting a stray line elsewhere in the photo (e.g. a page
    edge or header text) whose position doesn't fit the grid's regular
    spacing at all.

    The base unit is the *median* gap, not the minimum. A single spurious
    near-duplicate line close to the border (one _merge_nearby_hough_lines
    didn't quite catch) produces one anomalously small gap; using that as
    the base unit corrupts the multiple-of-base-unit check for every other
    gap and throws away most of the real grid lines (confirmed on image50/
    image32: min-based base unit kept only 4-8 of the 10-12 detected lines,
    median-based kept 9-11). The median is far more robust to one outlier
    since real grids have many same-sized gaps and at most a few 2x/3x ones.
    """
    if len(lines_with_position) < 3:
        return lines_with_position

    lines_with_position = sorted(lines_with_position, key=lambda item: item[1])
    positions = [position for _, position in lines_with_position]
    gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    base_unit = statistics.median(gaps)

    def is_valid_multiple(gap: float, max_multiple: int = 3) -> bool:
        if base_unit <= 0:
            return False
        nearest_multiple = round(gap / base_unit)
        return (
            1 <= nearest_multiple <= max_multiple
            and abs(gap / base_unit - nearest_multiple) <= tolerance
        )

    best_start, best_end = 0, 1
    start = 0
    for i, gap in enumerate(gaps):
        if not is_valid_multiple(gap):
            start = i + 1
        if i + 1 - start > best_end - best_start:
            best_start, best_end = start, i + 1

    return lines_with_position[best_start:best_end + 1]


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
