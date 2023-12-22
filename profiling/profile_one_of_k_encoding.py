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


from tqdm import tqdm
from graphium.utils.tensor import one_of_k_encoding


def main():
    CLASSES = ["AA", "BB", "CC", "DD", "EE"]
    CHOICES = CLASSES + ["FF"]

    for ii in tqdm(range(500000)):
        for choice in CHOICES:
            one_of_k_encoding(choice, CLASSES)

    print("DONE :)")


if __name__ == "__main__":
    main()
