import accuracy_reporter


def test_parse_puzzle_file_returns_9x9_grid_of_digit_strings(image1081_dat):
    grid = accuracy_reporter.parse_puzzle_file(image1081_dat)

    assert len(grid) == 9
    assert all(len(row) == 9 for row in grid)
    assert grid[0] == ["0", "0", "6", "0", "0", "2", "0", "3", "0"]


def test_calculate_accuracy_perfect_match_is_81():
    grid = [["1"] * 9 for _ in range(9)]

    accuracy = accuracy_reporter.calculate_accuracy(grid, grid, show_per_digit_stats=False)

    assert accuracy == 81


def test_calculate_accuracy_no_match_is_zero():
    true_digits = [["1"] * 9 for _ in range(9)]
    pred_digits = [["2"] * 9 for _ in range(9)]

    accuracy = accuracy_reporter.calculate_accuracy(true_digits, pred_digits, show_per_digit_stats=False)

    assert accuracy == 0
