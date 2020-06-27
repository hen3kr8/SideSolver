import argparse
import puzzle_extractor


def run(args):
    puzzle_extractor.main(args["puzzle_image"], args["puzzle_solution"], args["model"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i",
        "--puzzle_image",
        required=False,
        default="../test/image1081.jpg",
        help="path to input puzzle_image",
    )
    ap.add_argument(
        "-s",
        "--puzzle_solution",
        required=False,
        default="../test/image1081.dat",
        help="path to input puzzle_solution",
    )
    ap.add_argument(
        "-m",
        "--model",
        type=str,
        default="models/finalized_model.sav",
        help="path to model to be used",
    )
    args = vars(ap.parse_args())
    run(args)
