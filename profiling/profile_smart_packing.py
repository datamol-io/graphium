import tqdm
from goli.ipu.ipu_dataloader import smart_packing, fast_packing
import numpy as np


def main():
    pack_sizes = np.random.randint(5, 50, size=5000)
    for ii in tqdm.tqdm(range(10)):
        smart_packing(pack_sizes, batch_size=20)

    for ii in tqdm.tqdm(range(1000)):
        fast_packing(pack_sizes, batch_size=20)


if __name__ == "__main__":
    main()
