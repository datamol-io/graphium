from goli.config._loader import merge_dicts
import unittest as ut

class TestLoader(ut.TestCase):
    def test_merge_dicts(self):
        dict_a = {
            'a': {
                'b': {
                    'c': 1,
                    'd': 2
                },
                'e': 3
            },
            'f': 4
        }

        dict_b = {
            'a': {
                'b': {
                    'c': 5,
                    'g': 6
                }
            },
            'h': 7
        }

        expected_result = {
            'a': {
                'b': {
                    'c': 1,
                    'd': 2,
                    'g': 6
                },
                'e': 3
            },
            'f': 4,
            'h': 7
        }

        merge_dicts(dict_a, dict_b)

        assert dict_a == expected_result

# Main
if __name__ == '__main__':
    ut.main()