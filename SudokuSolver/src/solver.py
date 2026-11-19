# The following program is a sudoku solver I made to learn about back tracking,
# and because coding is fun.
# It is based of a video I saw on Numberphile.
# I own nothing. Please don't sue.
import numpy as np

puzzle = np.matrix(data=None)
puzzle_2 = np.matrix(data=None)


def read_puzzle():
    # global puzzle
    rel_path = "../sudoku_dataset-master/images"
    im1 = "image1.dat"
    im_path = rel_path + "/" + im1

    puzzle = np.loadtxt(im_path, skiprows=2, dtype=np.int)
    pretty_print_puzzle(puzzle)
    return puzzle


def solve():
    global puzzle
    global puzzle_2

    puzzle = read_puzzle()
    fill_in()
    # print("puzzle_1",puzzle_1.shape, type(puzzle_1))
    # print("puzzle_2", puzzle_2)
    # pretty_print_puzzle(puzzle_2)


def fill_in():
    global puzzle
    global puzzle_2

    n = puzzle.shape[0]

    for i in range(0, n):
        for j in range(0, n):
            if puzzle[i][j] == 0:
                for num in range(1, 10):
                    if valid_number(puzzle, i, j, num):
                        puzzle[i][j] = num
                        fill_in()
                        puzzle[i][j] = 0

                return
    pretty_print_puzzle(puzzle)


def valid_number(puzzle, x, y, num):
    n = puzzle.shape[0]
    cell_size = int(n ** 0.5)

    assert num != 0
    # valid row
    for i in range(0, n):
        if puzzle[i][y] == num:
            return False

    # valid column
    for i in range(0, n):
        if puzzle[x][i] == num:
            return False

    # 3 x 3 block - i don't believe converting to in is nec due to //?
    b_x = int(x // cell_size) * cell_size
    b_y = int(y // cell_size) * cell_size

    for i in range(0, 3):
        for j in range(0, 3):
            if puzzle[int(b_x) + i][int(b_y) + j] == num:
                return False

    return True


def solve_puzzle(initial_puzzle):
    """Solve a given 9x9 puzzle and return the solved grid.

    Unlike fill_in()/solve(), this does not rely on global state and stops
    as soon as a solution is found (instead of backtracking through all of
    them), so the returned grid is usable by callers.

    Uses bitmasks (one bit per digit 1-9) per row/column/box for O(1)
    candidate lookup instead of re-scanning the row/column/box on every
    placement attempt, and always branches on the blank cell with the
    fewest remaining candidates (MRV) instead of a fixed top-left-to-
    bottom-right order. Together these cut a real puzzle that took ~24s
    with the naive left-to-right/top-to-bottom backtracking down to well
    under a second, since most cells end up with only one or two
    candidates once a handful of digits are placed, and MRV finds those
    forced cells first instead of grinding through unconstrained ones.

    Arguments:
        initial_puzzle {array-like} -- 9 x 9 grid, 0 (or "0") for blank cells

    Returns:
        [np.array] -- solved 9 x 9 grid, or None if no solution exists.
    """

    puzzle = np.array(initial_puzzle, dtype=int)

    if not _givens_are_consistent(puzzle):
        return None

    if _fill_in_puzzle(puzzle):
        return puzzle

    return None


def _givens_are_consistent(puzzle):
    """Check that the non-blank cells don't already conflict with each other.

    valid_number() only ever runs when placing a *new* digit into a blank
    cell, so a conflict between two pre-filled givens (e.g. from a
    misclassified digit) would otherwise never be caught, and the grid would
    be returned as "solved" unchanged.
    """

    n = puzzle.shape[0]
    cell_size = int(n ** 0.5)

    for i in range(n):
        if not _no_duplicates(puzzle[i, :]):
            return False

    for j in range(n):
        if not _no_duplicates(puzzle[:, j]):
            return False

    for box_row in range(0, n, cell_size):
        for box_col in range(0, n, cell_size):
            box = puzzle[box_row:box_row + cell_size, box_col:box_col + cell_size]
            if not _no_duplicates(box.flatten()):
                return False

    return True


def _no_duplicates(values):
    non_blank = [v for v in values if v != 0]
    return len(non_blank) == len(set(non_blank))


def _fill_in_puzzle(puzzle):
    n = puzzle.shape[0]
    cell_size = int(n ** 0.5)
    full_mask = (1 << n) - 1

    def box_index(i, j):
        return (i // cell_size) * cell_size + (j // cell_size)

    row_mask = [0] * n
    col_mask = [0] * n
    box_mask = [0] * n
    blanks = []

    for i in range(n):
        for j in range(n):
            value = int(puzzle[i][j])
            if value == 0:
                blanks.append((i, j))
            else:
                bit = 1 << (value - 1)
                row_mask[i] |= bit
                col_mask[j] |= bit
                box_mask[box_index(i, j)] |= bit

    def candidates(i, j):
        used = row_mask[i] | col_mask[j] | box_mask[box_index(i, j)]
        return full_mask & ~used

    def fill(remaining_blanks):
        if not remaining_blanks:
            return True

        # MRV: branch on the blank cell with the fewest remaining
        # candidates, so cells with only one possible digit get resolved
        # (and propagate new constraints) before less-constrained ones.
        best_index, best_candidates, best_count = None, 0, n + 1
        for index, (i, j) in enumerate(remaining_blanks):
            cell_candidates = candidates(i, j)
            count = bin(cell_candidates).count("1")
            if count < best_count:
                best_index, best_candidates, best_count = index, cell_candidates, count
                if count <= 1:
                    break

        if best_count == 0:
            return False

        i, j = remaining_blanks[best_index]
        rest = remaining_blanks[:best_index] + remaining_blanks[best_index + 1:]
        b = box_index(i, j)

        remaining_candidates = best_candidates
        while remaining_candidates:
            bit = remaining_candidates & -remaining_candidates
            remaining_candidates ^= bit

            puzzle[i][j] = bit.bit_length()
            row_mask[i] |= bit
            col_mask[j] |= bit
            box_mask[b] |= bit

            if fill(rest):
                return True

            puzzle[i][j] = 0
            row_mask[i] &= ~bit
            col_mask[j] &= ~bit
            box_mask[b] &= ~bit

        return False

    return fill(blanks)

def pretty_print_puzzle(puzzle):

    puzzle = np.asarray(puzzle)
    n = puzzle.shape[0]

    print("  _ _ _   _ _ _   _ _ _")

    for i in range(0, n):

        for j in range(0, n):
            if j % (n ** 0.5) == 0:
                print("|", end=" ")

            print(int(puzzle[i][j]), end=" ")

        if i % (n ** 0.5) == (n ** 0.5) - 1:
            print("\n  _ _ _   _ _ _   _ _ _")

        print("")


if __name__ == "__main__":
    solve()
