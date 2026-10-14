import argparse
import os
import puzzle_extractor
import digit_classifier

_here = os.path.dirname(os.path.abspath(__file__))



def run(args):
    puzzle_extractor.main(args["puzzle_image"], args["puzzle_solution"], args["model"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i",
        "--puzzle_image",
        required=False,
        default=os.path.join(_here, "..", "test", "image1081.jpg"),
        help="path to input puzzle_image",
    )
    ap.add_argument(
        "-s",
        "--puzzle_solution",
        required=False,
        default=os.path.join(_here, "..", "test", "image1081.dat"),
        help="path to input puzzle_solution",
    )
    ap.add_argument(
        "-m",
        "--model",
        type=str,
        default=os.path.join(_here, "models", "finalized_model.keras"),
        help="path to model to be used",
    )
    args = vars(ap.parse_args())
    run(args)
