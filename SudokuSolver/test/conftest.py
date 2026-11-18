import os

import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_TEST_DIR, "..", "src")
_DATA_DIR = os.path.join(_TEST_DIR, "..", "data")


@pytest.fixture
def image1081_jpg() -> str:
    return os.path.join(_TEST_DIR, "image1081.jpg")


@pytest.fixture
def image1081_dat() -> str:
    return os.path.join(_TEST_DIR, "image1081.dat")


@pytest.fixture
def model_path() -> str:
    return os.path.join(_SRC_DIR, "models", "finalized_model.keras")


@pytest.fixture
def v2_test_raw_dir() -> str:
    return os.path.join(_DATA_DIR, "v2_test_raw")
