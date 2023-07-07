import pytest


@pytest.mark.gpu
def test_dummy_gpu():
    # This is a placeholder to validate the GPU test pipeline
    assert 1 + 1 == 2
