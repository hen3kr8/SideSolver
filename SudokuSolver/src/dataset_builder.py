import logging
import os

import accuracy_reporter
import cv2
import digit_extraction
import grid_detection
import matplotlib.pyplot as plt
import numpy as np


def create_digit_dataset(dataset_path: str, path_to_digit_folder: str):

    data_images = read_data_images(dataset_path)
    text_solutions = read_true_puzzle_digits_from_dir(dataset_path)
    logging.info("done reading data images and solutions from " + dataset_path)
    counter = 1
    for image_name, raw_image in data_images.items():

        thres_image = grid_detection.apply_threshold(raw_image)
        grid = grid_detection.find_largest_object(thres_image)
        corners = grid_detection.corner_detection(grid)

        if corners is None:
            logging.warning(
                "corners are None for image %s, skipping this image", image_name
            )
            continue


        # homography
        image_homog = grid_detection.apply_homography(raw_image, corners_src=corners)
        # extract digits
        image_digit_list = digit_extraction.extract_digits(image_homog)

        # process digits
        processed_image_digits = digit_extraction.process_digit_images_before_classification(
            image_digit_list
            )

        # actual digit list (so we know if the digit is 0, 1, 2, 3, 4, 5, 6, 7, 8, or 9)
        save_digits_to_folder(processed_image_digits, text_solutions[image_name], image_name, path_to_digit_folder)
        logging.info("done saving digits to folder for " + str(counter) + " of " + str(len(data_images)) + " images")
        counter += 1
    logging.info("done saving digits of all puzzles to folder")


def create_digit_folders(path_to_digit_folder: str = "../data/processed_digits/training"):

    # create folders 1-9
    directory = path_to_digit_folder
    for i in range(1,10):
        digit_directory = directory + "/" + str(i)
        if not os.path.isdir(digit_directory):
            os.makedirs(digit_directory, exist_ok=True)
    logging.info("done creating digit folders")


def save_digits_to_folder(processed_image_digits: np.ndarray, true_digits: list[list[str]], image_name: str, path_to_digit_folder: str):
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


def read_data_images(path: str) -> dict:
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


def read_true_puzzle_digits_from_dir(path: str) -> dict:
    """Read every text based sudoku solution in a directory, so we now can train our classifier, and split the datasets into digit folders.

    Arguments:
            path {str} -- path to training or test data text solutions

        Returns:
            [dict] -- dictionary of text solutions, where key is the image name and value is the text solution.
    """

    text_solutions = {}
    for file in os.listdir(path):
        if file.endswith(".dat"):
            file_name = file.split(".")[0]
            text_solutions[file_name] = accuracy_reporter.parse_puzzle_file(os.path.join(path, file))
    return text_solutions
