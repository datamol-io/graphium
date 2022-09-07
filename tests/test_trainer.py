"""
Unit tests for the metrics and wrappers of goli/utils/...
"""

import unittest as ut

from goli.utils.safe_run import SafeRun

class test_SafeRun(ut.TestCase):
    def test_safe_run(self):

        with SafeRun(name="bob", raise_error=False):
            raise ValueError("This is an error")

        with self.assertRaises(ValueError):
            with SafeRun(name="bob", raise_error=True):
                raise ValueError("This is an error")

        with SafeRun(name="bob", raise_error=True):
            print("This is not an error")

        with SafeRun(name="bob", raise_error=False):
            print("This is not an error")

if __name__ == "__main__":
    ut.main()
