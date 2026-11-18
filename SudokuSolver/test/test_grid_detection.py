import cv2
import grid_detection
import numpy as np


def test_merge_nearby_hough_lines_collapses_duplicates():
    # three lines close together (should merge into one) plus one far away
    lines_with_position = [
        (("line_a",), 10.0),
        (("line_b",), 11.5),
        (("line_c",), 12.0),
        (("line_d",), 100.0),
    ]

    merged = grid_detection._merge_nearby_hough_lines(lines_with_position, merge_dist=5)

    assert len(merged) == 2
    assert merged[0][1] == (10.0 + 11.5 + 12.0) / 3
    assert merged[1][1] == 100.0


def test_longest_regular_hough_run_tolerates_one_missing_line():
    # evenly spaced at ~50 apart, but one line (at ~150) is missing
    positions = [0.0, 50.0, 100.0, 200.0, 250.0, 300.0]
    lines_with_position = [((f"line_{i}",), p) for i, p in enumerate(positions)]

    run = grid_detection._longest_regular_hough_run(lines_with_position)

    assert [p for _, p in run] == positions


def test_longest_regular_hough_run_rejects_stray_outlier_before_the_grid():
    # one unrelated line (e.g. a page header) well above a coherent grid run
    # at ~50 spacing, which does not fit the regular spacing at all.
    positions = [10.0, 500.0, 550.0, 600.0, 650.0, 700.0, 750.0]
    lines_with_position = [((f"line_{i}",), p) for i, p in enumerate(positions)]

    run = grid_detection._longest_regular_hough_run(lines_with_position)

    assert [p for _, p in run] == [500.0, 550.0, 600.0, 650.0, 700.0, 750.0]


def test_longest_regular_hough_run_rejects_stray_outlier_after_the_grid():
    positions = [500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 1000.0]
    lines_with_position = [((f"line_{i}",), p) for i, p in enumerate(positions)]

    run = grid_detection._longest_regular_hough_run(lines_with_position)

    assert [p for _, p in run] == [500.0, 550.0, 600.0, 650.0, 700.0, 750.0]


def test_longest_regular_hough_run_sorts_unsorted_input():
    positions = [650.0, 500.0, 750.0, 550.0, 700.0, 600.0]
    lines_with_position = [((f"line_{i}",), p) for i, p in enumerate(positions)]

    run = grid_detection._longest_regular_hough_run(lines_with_position)

    assert [p for _, p in run] == sorted(positions)


def test_hough_line_intersection_of_perpendicular_lines():
    # vertical line x=5 (theta=0, rho=5) and horizontal line y=10 (theta=pi/2, rho=10)
    vertical_line = (5.0, 0.0)
    horizontal_line = (10.0, np.pi / 2)

    point = grid_detection._hough_line_intersection(vertical_line, horizontal_line)

    assert point is not None
    assert np.allclose(point, (5.0, 10.0))


def test_hough_line_intersection_returns_none_for_parallel_lines():
    line1 = (5.0, 0.0)
    line2 = (10.0, 0.0)

    assert grid_detection._hough_line_intersection(line1, line2) is None


def test_detect_grid_corners_hough_finds_a_plausible_grid(image1081_jpg):
    raw_image = grid_detection.read_image(image1081_jpg)

    corners = grid_detection.detect_grid_corners_hough(raw_image)

    assert corners is not None
    assert len(corners) == 4


def test_detect_grid_corners_hough_returns_none_for_blank_image():
    blank = np.full((400, 400), 200, dtype=np.uint8)

    assert grid_detection.detect_grid_corners_hough(blank) is None


def test_detect_grid_corners_falls_back_to_hough_when_floodfill_fails(monkeypatch, image1081_jpg):
    raw_image = grid_detection.read_image(image1081_jpg)

    monkeypatch.setattr(grid_detection, "corner_detection", lambda image: None)

    corners = grid_detection.detect_grid_corners(raw_image)

    assert corners is not None
