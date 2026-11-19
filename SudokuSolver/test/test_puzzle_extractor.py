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


@pytest.mark.slow
def test_full_dataset_accuracy_does_not_regress(v2_test_raw_dir, model_path):
    """Guards the real aggregate accuracy across all of v2_test_raw, not just one
    fixture image. image1081 alone hit 100%, but that hid two things this catches:
    corner detection failing on some photos entirely, and blank-cell detection
    being measurably worse than digit classification itself. Thresholds are set
    with headroom below the measured baseline so this only fails on an actual
    regression, not routine noise.

    The Hough fallback in detect_grid_corners() traded some of these numbers
    for far fewer skipped images: num_skipped dropped from 9 to 1 (then to 0
    after fixing _longest_regular_hough_run's base-unit calculation - see
    below), but non_blank_accuracy and overall_accuracy dropped too, because
    the rescued images are often imprecisely aligned. Verified separately
    that none of them produce a wrong "solved" result -- they all fail
    safely via solve_puzzle_from_image() returning None -- so this is a real
    but contained trade-off, not a silent correctness regression.

    _longest_regular_hough_run used min(gaps) as the base grid-cell-size
    unit, which a single spurious near-duplicate line near a border (one
    gap anomalously small) would corrupt, throwing out most of the real
    grid lines (confirmed on image50/image32). Switched to median(gaps),
    which recovers the real lines instead. Net effect: num_skipped 1 -> 0
    (image51 recovered), 3 of the other rescued images improved, none
    regressed.

    Separately, is_blank_digit's threshold was tuned from 1/40 to 1/22 (see
    TODO.md) -- raises overall_accuracy and perfect_puzzle_rate on top of the
    Hough changes above (non_blank_accuracy is untouched by that fix, since
    it only affects blank-vs-digit decisions). Measured with both fixes
    together on the full v2_test_raw set: num_skipped 0, non_blank_accuracy
    0.7751, overall_accuracy 0.8821, perfect_puzzle_rate 0.475. Thresholds
    below keep headroom under those numbers.
    """

    results = puzzle_extractor.evaluate_dataset_accuracy(v2_test_raw_dir, model_path)

    assert results["num_skipped"] <= 1
    assert results["non_blank_accuracy"] >= 0.73
    assert results["overall_accuracy"] >= 0.84
    assert results["perfect_puzzle_rate"] >= 0.40
