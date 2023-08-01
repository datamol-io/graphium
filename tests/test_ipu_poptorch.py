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
