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

    Arguments:
        initial_puzzle {array-like} -- 9 x 9 grid, 0 (or "0") for blank cells

    Returns:
        [np.array] -- solved 9 x 9 grid, or None if no solution exists.
    """

    puzzle = np.array(initial_puzzle, dtype=int)

    if _fill_in_puzzle(puzzle):
        return puzzle

    return None


def _fill_in_puzzle(puzzle):
    n = puzzle.shape[0]

    for i in range(0, n):
        for j in range(0, n):
            if puzzle[i][j] == 0:
                for num in range(1, 10):
                    if valid_number(puzzle, i, j, num):
                        puzzle[i][j] = num
                        if _fill_in_puzzle(puzzle):
                            return True
                        puzzle[i][j] = 0

                return False

    return True

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
