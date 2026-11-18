"""Regression tests that run the real extraction/classification/solving
pipeline against the known-good image1081 fixture.

These are integration tests, not unit tests: they load the real TensorFlow
model and run actual image processing, so they're slower than the rest of
the suite. They exist specifically to catch regressions like the
predict_number off-by-one bug, where every individual function looked fine
in isolation but the end-to-end result was wrong.
"""

import accuracy_reporter
import digit_extraction
import grid_detection
import numpy as np
import puzzle_extractor
import pytest


def test_image1081_classifies_with_perfect_accuracy(image1081_jpg, image1081_dat, model_path):
    raw_image = grid_detection.read_image(image1081_jpg)
    thres_image = grid_detection.apply_threshold(raw_image)
    grid = grid_detection.find_largest_object(thres_image)
    corners = grid_detection.corner_detection(grid)
    assert corners is not None

    image_homog = grid_detection.apply_homography(raw_image, corners_src=corners)
    image_digit_list = digit_extraction.extract_digits(image_homog)
    processed_image_digits = digit_extraction.process_digit_images_before_classification(
        image_digit_list
    )

    classified_digits = puzzle_extractor.classify_image_digits(processed_image_digits, model_path)

    true_digits = accuracy_reporter.parse_puzzle_file(image1081_dat)
    accuracy = accuracy_reporter.calculate_accuracy(true_digits, classified_digits, show_per_digit_stats=False)

    assert accuracy == 81


@pytest.mark.slow
def test_image1081_solves_end_to_end(image1081_jpg, model_path):
    solved_image = puzzle_extractor.solve_puzzle_from_image(image1081_jpg, model_path)

    assert solved_image is not None
    assert isinstance(solved_image, np.ndarray)
    assert solved_image.ndim == 3  # BGR overlay image
