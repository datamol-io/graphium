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
