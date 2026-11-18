import accuracy_reporter
import config
import cv2
import dataset_builder
import digit_classifier
import digit_extraction
import grid_detection
import logging
import matplotlib.pyplot as plt
import numpy as np
import solver


create_training_set = False
create_test_set = False
training_set_path = "../data/v2_train_raw/image108"
test_set_path = "../data/v2_test_raw"
path_to_processed_training_digits_directory = "../data/processed_digits/training"
path_to_processed_test_digits_directory = "../data/processed_digits/test"


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def main(
    puzzle_image: str = "../data/v2_train_raw/image1081.jpg",
    puzzle_solution: str = "../data/v2_train_raw/image1081.dat",
    # puzzle_image="../sudoku_dataset-master/images/image1081.jpg",
    # puzzle_solution="../sudoku_dataset-master/images/image1081.dat",
    model: str = "models/finalized_model.keras",
):


    if create_training_set:
        dataset_builder.create_digit_folders(path_to_digit_folder=path_to_processed_training_digits_directory)
        logging.info("creating training digit dataset")
        dataset_builder.create_digit_dataset(training_set_path, path_to_digit_folder=path_to_processed_training_digits_directory)
        logging.info("done creating training digit dataset")

    if create_test_set:
        dataset_builder.create_digit_folders(path_to_digit_folder=path_to_processed_test_digits_directory)
        logging.info("creating test digit dataset")
        dataset_builder.create_digit_dataset(test_set_path, path_to_digit_folder=path_to_processed_test_digits_directory)
        logging.info("done creating test digit dataset")

    else:

        logging.info("skipping digit dataset creation")
        extract_digits_from_image(puzzle_image, puzzle_solution, model)
        solve_puzzle_from_image(puzzle_image, model)


def extract_digits_from_image(puzzle_image: str, puzzle_solution: str, model: str):
    raw_image = grid_detection.read_image(puzzle_image)
    corners = grid_detection.detect_grid_corners(raw_image)

    if corners is None:
        logging.warning(
            "corners could not be detected for %s, skipping this image", puzzle_image
        )
        return

    # homography
    image_homog = grid_detection.apply_homography(raw_image, corners_src=corners)

    # extract digits
    image_digit_list = digit_extraction.extract_digits(image_homog)

    # process digits
    processed_image_digits = digit_extraction.process_digit_images_before_classification(
        image_digit_list
    )

    # classify digits
    classified_digits = classify_image_digits(processed_image_digits, model)

    # accuracy of puzzle
    true_digits = accuracy_reporter.read_true_puzzle_digits(puzzle_solution)
    accuracy_reporter.calculate_accuracy(true_digits, classified_digits, False)


def evaluate_dataset_accuracy(dataset_path: str, model: str) -> dict:
    """Run extraction + classification (no solving) over every labeled image in
    dataset_path and report aggregate accuracy, instead of just one fixture image.

    Arguments:
        dataset_path {str} -- directory of raw labeled puzzle photos (.jpg + .dat pairs)
        model {str} -- path to the digit classifier model

    Returns:
        [dict] -- aggregate accuracy stats across the whole dataset
    """

    data_images = dataset_builder.read_data_images(dataset_path)
    text_solutions = dataset_builder.read_true_puzzle_digits_from_dir(dataset_path)

    total_correct = 0
    total_non_blank = 0
    total_non_blank_correct = 0
    perfect_puzzles = 0
    skipped_images = []
    per_image_accuracy = {}

    image_names = sorted(data_images.keys())
    for i, image_name in enumerate(image_names, start=1):
        logging.info("evaluating %s (%d of %d)", image_name, i, len(image_names))
        raw_image = data_images[image_name]

        corners = grid_detection.detect_grid_corners(raw_image)

        if corners is None:
            logging.warning("corners not found for %s, skipping", image_name)
            skipped_images.append(image_name)
            continue

        image_homog = grid_detection.apply_homography(raw_image, corners_src=corners)
        image_digit_list = digit_extraction.extract_digits(image_homog)
        processed_image_digits = digit_extraction.process_digit_images_before_classification(
            image_digit_list
        )
        classified_digits = classify_image_digits(processed_image_digits, model)

        true_flat = [int(d) for row in text_solutions[image_name] for d in row]
        pred_flat = [int(d) for row in classified_digits for d in row]

        correct = sum(t == p for t, p in zip(true_flat, pred_flat))
        non_blank = sum(t != 0 for t in true_flat)
        non_blank_correct = sum(
            t == p for t, p in zip(true_flat, pred_flat) if t != 0
        )

        total_correct += correct
        total_non_blank += non_blank
        total_non_blank_correct += non_blank_correct
        if correct == 81:
            perfect_puzzles += 1

        per_image_accuracy[image_name] = correct / 81.0

    num_evaluated = len(image_names) - len(skipped_images)
    total_positions = num_evaluated * 81

    results = {
        "num_images": len(image_names),
        "num_evaluated": num_evaluated,
        "num_skipped": len(skipped_images),
        "skipped_images": skipped_images,
        "overall_accuracy": total_correct / total_positions if total_positions else 0.0,
        "non_blank_accuracy": total_non_blank_correct / total_non_blank if total_non_blank else 0.0,
        "perfect_puzzles": perfect_puzzles,
        "perfect_puzzle_rate": perfect_puzzles / num_evaluated if num_evaluated else 0.0,
        "per_image_accuracy": per_image_accuracy,
    }

    logging.info("=== Full dataset accuracy report (%s) ===", dataset_path)
    logging.info(
        "Images: %d total, %d evaluated, %d skipped (no grid detected): %s",
        results["num_images"], num_evaluated, results["num_skipped"], skipped_images,
    )
    logging.info("Overall positional accuracy: %.4f", results["overall_accuracy"])
    logging.info("Non-blank digit accuracy: %.4f", results["non_blank_accuracy"])
    logging.info(
        "Fully-correct puzzles: %d / %d (%.4f)",
        perfect_puzzles, num_evaluated, results["perfect_puzzle_rate"],
    )

    return results


def solve_puzzle_from_image(puzzle_image: str, model: str):
    """Extract, classify and solve a sudoku puzzle from an image, then draw
    the solved digits on top of the blank cells.

    Arguments:
        puzzle_image {str} -- path to the puzzle image
        model {str} -- path to the digit classifier model

    Returns:
        [np.array] -- image with the solved digits drawn on the blank cells,
                      or None if the grid corners could not be detected or
                      no solution was found.
    """

    raw_image = grid_detection.read_image(puzzle_image)
    corners = grid_detection.detect_grid_corners(raw_image)

    if corners is None:
        logging.warning(
            "corners could not be detected for %s, cannot solve this image", puzzle_image
        )
        return None

    # homography
    image_homog = grid_detection.apply_homography(raw_image, corners_src=corners)

    # extract digits
    image_digit_list = digit_extraction.extract_digits(image_homog)

    # process digits
    processed_image_digits = digit_extraction.process_digit_images_before_classification(
        image_digit_list
    )

    # classify digits
    classified_digits = classify_image_digits(processed_image_digits, model)

    # solve puzzle
    solved_digits = solver.solve_puzzle(classified_digits)

    if solved_digits is None:
        logging.warning("could not solve puzzle - no solution found.")
        return None

    solved_image = draw_solution_on_image(image_homog, classified_digits, solved_digits)

    if config.display_images_flag:
        plt.imshow(solved_image)
        plt.title("solved puzzle")
        plt.show()

    return solved_image


def draw_solution_on_image(image_homog: np.ndarray, classified_digits: np.ndarray, solved_digits: np.ndarray):
    """Draw the solved digit on top of each cell that was originally blank.

    Arguments:
        image_homog {np.array} -- straightened grayscale image of the puzzle
        classified_digits {np.array} -- 9 x 9 grid of digits recognized from
                                         the image (0 for blank cells)
        solved_digits {np.array} -- 9 x 9 grid of the solved puzzle

    Returns:
        [np.array] -- BGR image with the solved digits drawn on blank cells
    """

    solved_image = cv2.cvtColor(image_homog.astype("uint8"), cv2.COLOR_GRAY2BGR)
    h, w = image_homog.shape
    cell_h, cell_w = h / 9, w / 9
    font_scale = min(cell_h, cell_w) / 40

    for i in range(0, 9):
        for j in range(0, 9):
            if int(classified_digits[i][j]) == 0:
                digit_text = str(int(solved_digits[i][j]))
                text_size, _ = cv2.getTextSize(
                    digit_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
                )
                text_x = int((j + 0.5) * cell_w - text_size[0] / 2)
                text_y = int((i + 0.5) * cell_h + text_size[1] / 2)

                cv2.putText(
                    solved_image,
                    digit_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

    return solved_image


def classify_image_digits(processed_image_digits: np.ndarray, model: str):
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


if __name__ == "__main__":
    main()
