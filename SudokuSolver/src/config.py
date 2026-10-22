import sys

import matplotlib
import numpy as np

matplotlib.use("QtAgg")

display_images_flag = False
debug = False

if debug:
    np.set_printoptions(threshold=sys.maxsize)
