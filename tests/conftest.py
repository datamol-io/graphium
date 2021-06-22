import pathlib

import pytest

TEST_DIR_PATH = pathlib.Path(__file__).parent / "data"
DATA_DIR_PATH = TEST_DIR_PATH.absolute()


@pytest.fixture
def datadir(request):
    return DATA_DIR_PATH
