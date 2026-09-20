import accuracy_reporter
import cv2
import digit_classifier
import logging
import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np
import operator
import solver
import sys
import os 


display_images_flag = False
debug = False
create_training_set = False
create_test_set = False
training_set_path = "../data/v2_train_raw/"
test_set_path = "../data/v2_test_raw"
path_to_processed_training_digits_directory = "../data/processed_digits/training"
path_to_processed_test_digits_directory = "../data/processed_digits/test"


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

if debug:
    np.set_printoptions(threshold=sys.maxsize)


def main(
    puzzle_image="../test/image1081.jpg",
    puzzle_solution="../test/image1081.dat",
    # puzzle_image="../sudoku_dataset-master/images/image1081.jpg",
    # puzzle_solution="../sudoku_dataset-master/images/image1081.dat",
    model="models/finalized_model.keras",
):

    
    if create_training_set:
        create_digit_folders(path_to_digit_folder=path_to_processed_training_digits_directory)
        logging.info("creating training digit dataset")
        create_digit_dataset(training_set_path, path_to_digit_folder=path_to_processed_training_digits_directory)
        logging.info("done creating training digit dataset")

    if create_test_set:
        create_digit_folders(path_to_digit_folder=path_to_processed_test_digits_directory)
        logging.info("creating test digit dataset")
        create_digit_dataset(test_set_path, path_to_digit_folder=path_to_processed_test_digits_directory)
        logging.info("done creating test digit dataset")

    else:

        logging.info("skipping digit dataset creation")
        raw_image = read_image(puzzle_image)
        thres_image = apply_threshold(raw_image)
        grid = find_largest_object(thres_image)
        corners = corner_detection(grid)

        # if display_images_flag:
        #     plot_corners_original(raw_image, corners)

        # homography
        image_homog = apply_homography(raw_image, corners_src=corners)

        # extract digits
        image_digit_list = extract_digits(image_homog)

        # process digits
        processed_image_digits = process_digit_images_before_classification(
            image_digit_list
        )

        # classify digits
        # classified_digits = classify_image_digits(processed_image_digits, model)

        # accuracy of puzzle
        # true_digits = accuracy_reporter.read_true_puzzle_digits(puzzle_solution)
        # accuracy_reporter.calculate_accuracy(true_digits, classified_digits)


def create_digit_dataset(dataset_path, path_to_digit_folder):

    data_images = read_data_images(dataset_path)
    text_solutions = read_true_puzzle_digits(dataset_path)
    logging.info("done reading data images and solutions from " + dataset_path)
    counter = 1
    for image_name, raw_image in data_images.items():

        thres_image = apply_threshold(raw_image)
        grid = find_largest_object(thres_image)
        corners = corner_detection(grid)

        if corners is None:
            logging.warning(
                "corners are None for image %s, skipping this image", image_name
            )
            continue


        # homography
        image_homog = apply_homography(raw_image, corners_src=corners)
        # if display_images_flag and image_name in ("image73", "image35", "image114", "image1081", "image117"):
        #     plt.imshow(image_homog, cmap="gray")
        #     plt.title("image_homog")
        #     plt.show()
        # extract digits
        image_digit_list = extract_digits(image_homog)

        # process digits
        processed_image_digits = process_digit_images_before_classification(
            image_digit_list
            )

        # actual digit list (so we know if the digit is 0, 1, 2, 3, 4, 5, 6, 7, 8, or 9)
        save_digits_to_folder(processed_image_digits, text_solutions[image_name], image_name, path_to_digit_folder)
        logging.info("done saving digits to folder for " + str(counter) + " of " + str(len(data_images)) + " images")
        counter += 1
    logging.info("done saving digits of all puzzles to folder")


def create_digit_folders(path_to_digit_folder="../data/processed_digits/training"):

    # create folders 1-9
    directory = path_to_digit_folder
    for i in range(1,10):
        digit_directory = directory + "/" + str(i)
        if not os.path.isdir(digit_directory):
            os.makedirs(digit_directory, exist_ok=True)
    logging.info("done creating digit folders")


def save_digits_to_folder(processed_image_digits, true_digits, image_name, path_to_digit_folder):
    """Save the processed digit images to folders, so that they can be used to train the classifier.

    Arguments:
        processed_image_digits {list of list of np.array} -- 9 x 9 list of lists, each
                                                       containing an image of a digit
        true_digits {list of list of int} -- 9 x 9 list of lists, each
                                                       containing the true digit
        image_name {str} -- name of the image, used to create unique folder names.
        path_to_digit_folder {str} -- path to the folder where the digits will be saved.
    """

    display_images_flag_local = False
    
    logging.info("saving digits to folder")
    for i in range(0,9):
        for j in range(0,9):
            true_digit = true_digits[i][j]
            if true_digit != "0" and isinstance(processed_image_digits[i][j], np.ndarray):
                # digit is not blank according to label, and we did not identify it as blank, so we save it to the folder.
                if display_images_flag_local:
                    plt.imshow(processed_image_digits[i][j], cmap="gray")
                    plt.title("digit " + str(true_digit))
                    plt.show()

                digit_path = path_to_digit_folder + "/" + str(true_digit) + "/" + image_name + str(i) + '_' + str(j) + ".png"
                cv2.imwrite(digit_path, processed_image_digits[i][j])
    logging.info("done saving digits to folder for image " + image_name)
    

def read_data_images(path)-> dict:
    """Read training and test data images from path (2 different folders). 
    This is used to create the digits which later will train and testthe classifier. 

    Arguments:
        path {str} -- path to training or test data images

    Returns:
        [dict] -- dictionary of images, where key is the image name and value is the image itself.
    """
    data_images = {}
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            # image = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
            image_name = file.split(".")[0]
            data_images[image_name] = image
    return data_images


def read_true_puzzle_digits(path) -> dict:
    """Read the text based sudoku, so we now can train our classifier, and split the datasets into digit folders.

    Arguments:
            path {str} -- path to training or test data text solutions
    
        Returns:
            [dict] -- dictionary of text solutions, where key is the image name and value is the text solution.
    """


    text_solutions = {}
    for file in os.listdir(path):
        if file.endswith(".dat"):
            text_solution = [i.strip().split() for i in open(os.path.join(path, file)).readlines()[2:]]
            file_name = file.split(".")[0]
            text_solutions[file_name] = text_solution
    # need to check - might have to cvert to numpy array.
    print("text_solutions: ", text_solutions)
    return text_solutions


def read_image(puzzle_image):

    # file_name = "../sudoku_dataset-master/images/image1024.jpg"

    image = cv2.imread(puzzle_image, cv2.IMREAD_GRAYSCALE)

    if display_images_flag:

        plt.imshow(image, cmap="gray")
        plt.title("original image")
        plt.show()

    return image


def blur_image(src_image):

    blurred_image = cv2.GaussianBlur(src_image, (15, 15), 1)

    if display_images_flag:
        plt.imshow(blurred_image, cmap="gray")
        plt.title("blurred image")
        plt.show()

    return blurred_image


def apply_threshold(src_image, bin=False):

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
    if display_images_flag:
        plt.imshow(thres_image, cmap="gray")
        plt.title("thresholded image 11")
        plt.show()

    return thres_image


def find_largest_object(image):
    """apply a blob detecting algorithm. In this case floodfilling.

    """

    new_image = flood_filling(image)
    grid_image = find_biggest_blob(new_image)

    if display_images_flag:
        plt.imshow(grid_image, cmap="gray")
        plt.title("Extracted grid")
        plt.show()

    return grid_image


def flood_filling(image):
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


def find_biggest_blob(new_image, largest_island=2):

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


def search(i, j, s, h, w):
    if i > 0:     s.append((i - 1, j))
    if i < h - 1: s.append((i + 1, j))
    if j < w - 1: s.append((i, j + 1))
    if j > 0:     s.append((i, j - 1))


def corner_detection(image):

    # The picture has to be in uint8 format. It was in float64.
    image_contour = image.astype("uint8") * 255

    contours, _ = cv2.findContours(
        image_contour, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
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

    if display_images_flag:
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
    ratio = quad_area / blob_area

    if ratio > 1.4: # some corners are not detected correctly, so we skip this image. The ratio of the area of the quad to the area of the blob should be less than 1.4.
        logging.warning(
            "The ratio of the area of the quad to the area of the blob is %s, which is greater than 1.4. This might indicate that the corners are not detected correctly.",
            ratio,
        )
        return None

    return corners


def plot_corners_original(image, corners):
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


def apply_homography(raw_image, corners_src=None, corners_dst=None, new_image=None):
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

    if display_images_flag:
        plt.imshow(image_homog, cmap="gray")
        plt.title("applied homography")
        plt.show()

    logging.debug("applied homography")
    return image_homog


def crop_center_image(image):
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

    if debug:
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

    if display_images_flag:
        plt.imshow(cropped_image)
        plt.title("cropped_image")
        plt.show()

    return cropped_image


def remove_noise(image_digit):
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

    if display_images_flag:
        plt.imshow(noise_free_image, cmap="gray")
        plt.title("removed all noise before prediction")
        plt.show()

    noise_free_image = np.invert(noise_free_image)
    return noise_free_image


def is_blank_digit(image_digit):
    """Determine whether image is a digit or blank.

    Apply flood filling, if the largest object is less than 1/10 of the
    total image, then it is considered blank.

    Arguments:
        image_digit {np.array} -- image of digit

    Returns:
        [Boolean]
    """

    new_image = flood_filling(image_digit)
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


def reshape_digit_image(image, new_image_shape=(32, 32), margin=4):
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




def extract_digits(image_homog):
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


def process_digit_images_before_classification(image_digit_list):
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
            thres_image = apply_threshold(cropped_image)
            digit = find_largest_object(thres_image)

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


def classify_image_digits(processed_image_digits, model):
    predicted_digits = np.zeros((9, 9))
    loaded_model = digit_classifier.load_model(model)

    for i in range(0, 9):
        for j in range(0, 9):
            predicted_digit = 0
            digit_to_be_classified = processed_image_digits[i][j]

            if type(digit_to_be_classified) != int:  # non blank, need better method.
                predicted_digit = digit_classifier.predict_number(
                    digit_to_be_classified, loaded_model
                )

            predicted_digits[i][j] = predicted_digit

    solver.pretty_print_puzzle(predicted_digits)
    logging.info("done classifying all digits")

    return predicted_digits


def invert(image):
    return np.invert(image)


def center_image(flood_filled_image, cropped_image):
    # find center in floodfilled
    display_images_flag = False
    h, w = flood_filled_image.shape

    # the below code shows which rows and columns are completely blank and can
    # be removed
    rows_containing_digit = np.sum(flood_filled_image, axis=1)
    cols_containing_digit = np.sum(flood_filled_image, axis=0)

    new_image = np.zeros(shape=(flood_filled_image.shape))

    # map digit from original image to a new image (thus ignoring noise)
    # for i in range(0, h):
    #     for j in range(0, w):
    #         if flood_filled_image[i, j] == 255:
    #             new_image[i, j] = 255 #cropped_image[i, j]
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

    # add padding
    # row_padding = np.zeros(shape=(5, centered_image.shape[1]))
    # centered_image = np.concatenate((row_padding, centered_image), axis=0)
    # centered_image = np.concatenate((centered_image, row_padding), axis=0)

    # col_padding = np.zeros(shape=(centered_image.shape[0], 5))
    # centered_image = np.concatenate((col_padding, centered_image), axis=1)
    # centered_image = np.concatenate((centered_image, col_padding), axis=1)

    logging.debug("padding added")
    return centered_image


def calculate_accuracy():

    pass


if __name__ == "__main__":
    main()
