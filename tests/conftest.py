"""
--------------------------------------------------------------------------------
Copyright (c) 2023 Valence Labs, Recursion Pharmaceuticals.
Use of this software is subject to the terms and conditions outlined in the LICENSE file.
Unauthorized modification, distribution, or use is prohibited. Provided 'as is' without
warranties of any kind.

Valence Labs, Recursion Pharmaceuticals are not liable for any damages arising from its use.
Refer to the LICENSE file for the full terms and conditions.
--------------------------------------------------------------------------------
"""


import pathlib

import pytest

TEST_DIR_PATH = pathlib.Path(__file__).parent / "data"
DATA_DIR_PATH = TEST_DIR_PATH.absolute()


@pytest.fixture
def datadir(request):
    return DATA_DIR_PATH
