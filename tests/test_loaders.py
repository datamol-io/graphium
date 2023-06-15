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
