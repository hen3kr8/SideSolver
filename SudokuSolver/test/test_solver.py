import accuracy_reporter
import numpy as np
import solver

SOLVED_GRID = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
]


def _assert_valid_sudoku(grid):
    """A valid completed sudoku has 1-9 exactly once in every row, column and 3x3 box."""
    grid = np.asarray(grid, dtype=int)
    expected = set(range(1, 10))

    for row in grid:
        assert set(row.tolist()) == expected

    for col in grid.T:
        assert set(col.tolist()) == expected

    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            box = grid[box_row:box_row + 3, box_col:box_col + 3]
            assert set(box.flatten().tolist()) == expected


def test_solve_puzzle_solves_real_puzzle(image1081_dat):
    givens = accuracy_reporter.parse_puzzle_file(image1081_dat)

    solved = solver.solve_puzzle(givens)

    assert solved is not None
    _assert_valid_sudoku(solved)

    # the solver must not have touched any of the originally given digits
    givens_grid = np.array(givens, dtype=int)
    non_blank = givens_grid != 0
    assert np.array_equal(solved[non_blank], givens_grid[non_blank])


def test_solve_puzzle_already_solved_grid_is_returned_unchanged():
    solved = solver.solve_puzzle(SOLVED_GRID)

    assert solved is not None
    assert np.array_equal(solved, np.array(SOLVED_GRID))


def test_solve_puzzle_returns_none_when_unsolvable():
    # blank out a cell, then make the digit it needs (5) appear a second time
    # elsewhere in the same row, so no candidate can ever satisfy that cell.
    unsolvable = [row[:] for row in SOLVED_GRID]
    unsolvable[0][1] = 5
    unsolvable[0][0] = 0

    assert solver.solve_puzzle(unsolvable) is None


def test_solve_puzzle_returns_none_for_conflicting_givens():
    # fully filled, but two 5s in row 0 -- no blank cells, so the only way to
    # catch this is by validating the givens themselves up front.
    conflicting_givens = [row[:] for row in SOLVED_GRID]
    conflicting_givens[0][1] = 5

    assert solver.solve_puzzle(conflicting_givens) is None


def test_solve_puzzle_returns_none_for_conflicting_givens_in_column():
    # two 5s in column 0, different rows and boxes -- isolates the column check.
    grid = [[0] * 9 for _ in range(9)]
    grid[0][0] = 5
    grid[3][0] = 5

    assert solver.solve_puzzle(grid) is None


def test_solve_puzzle_returns_none_for_conflicting_givens_in_box():
    # two 5s in the top-left box, different rows and columns -- isolates the box check.
    grid = [[0] * 9 for _ in range(9)]
    grid[0][0] = 5
    grid[1][1] = 5

    assert solver.solve_puzzle(grid) is None
