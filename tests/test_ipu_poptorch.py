"""
--------------------------------------------------------------------------------
Copyright (c) 2023 Valence Labs, Recursion Pharmaceuticals and Graphcore Limited.

Use of this software is subject to the terms and conditions outlined in the LICENSE file.
Unauthorized modification, distribution, or use is prohibited. Provided 'as is' without
warranties of any kind.

Valence Labs, Recursion Pharmaceuticals and Graphcore Limited are not liable for any damages arising from its use.
Refer to the LICENSE file for the full terms and conditions.
--------------------------------------------------------------------------------
"""


import pytest


@pytest.mark.ipu
def test_poptorch():
    # Run this test only if poptorch is available
    # Primarily to test the install and SDK is correctly activated
    try:
        import poptorch

        opts = poptorch.Options()

    except ImportError:
        raise ImportError
    assert True
