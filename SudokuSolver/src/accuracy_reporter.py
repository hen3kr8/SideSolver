# Compare puzzle recognized by classifier to label (puzzle saved in dataset)
import logging
import solver

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def read_true_puzzle_digits(puzzle_solution):

    loc = puzzle_solution
    true_digits = [i.strip().split() for i in open(loc).readlines()[2:]]

    logging.info("REAL PUZZLE")
    solver.pretty_print_puzzle(true_digits)
    return true_digits


def calculate_accuracy(true_digits, pred_digits, show_per_digit_stats=True):

    overall_accuracy = 0
    occurence_per_digit = {i: 0 for i in range(0, 10)}
    accuracy_per_digit = {i: 0 for i in range(0, 10)}

    true_digits = [int(d) for row in true_digits for d in row]
    pred_digits = [int(d) for row in pred_digits for d in row]

    for i, j in zip(true_digits, pred_digits):
        if i == j:
            overall_accuracy += 1
            accuracy_per_digit[i] += 1
        occurence_per_digit[int(i)] += 1

    total_non_blank_digits = 81.0 - occurence_per_digit[0]
    blank_accuracy = accuracy_per_digit[0]

    logging.info("Accuracy (including blanks): %s ", overall_accuracy / 81.0)
    logging.info(
        "Accuracy (excluding blanks): %s ",
        (overall_accuracy - blank_accuracy) / total_non_blank_digits,
    )

    logging.info(
        "Positional Accuracy: %s of 81 positions classified correctly.",
        overall_accuracy,
    )
    logging.info(
        "Digit Classification Accuracy: %s of %s positions classified correctly.",
        overall_accuracy - blank_accuracy,
        81 - blank_accuracy,
    )

    if show_per_digit_stats:
        logging.info("accuracy per digit:")
        for d in accuracy_per_digit.keys():
            perc_accuracy_digit = accuracy_per_digit[d] / occurence_per_digit[d]
            print(
                d,
                " -- ",
                perc_accuracy_digit,
                " (",
                accuracy_per_digit[d],
                " / ",
                occurence_per_digit[d],
                ")",
            )

    return overall_accuracy
