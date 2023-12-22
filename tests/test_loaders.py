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


from graphium.config._loader import merge_dicts
from copy import deepcopy
import unittest as ut


class TestLoader(ut.TestCase):
    def test_merge_dicts(self):
        dict_a = {"a": {"b": {"c": 1, "d": 2}, "e": 3}, "f": 4}

        dict_b = {"a": {"b": {"g": 6}}, "h": 7}

        dict_c = {
            "a": {
                "b": {
                    "c": 1,
                },
            }
        }

        # Check that the new keys are added correctly
        merge_dicts(dict_a, dict_b)
        self.assertEqual(dict_a["a"]["b"]["g"], dict_b["a"]["b"]["g"])
        self.assertEqual(dict_a["h"], dict_b["h"])

        # Check that no error is thrown if a key exists, but the value is identical
        merge_dicts(dict_a, dict_c)
        self.assertEqual(dict_a["a"]["b"]["c"], dict_c["a"]["b"]["c"])

        # Check that an error is thrown if a key exists, but the value is different
        dict_d = deepcopy(dict_c)
        dict_d["a"]["b"]["c"] = 2
        with self.assertRaises(ValueError):
            merge_dicts(dict_a, dict_d)


# Main
if __name__ == "__main__":
    ut.main()
