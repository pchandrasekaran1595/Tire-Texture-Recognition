import os
import sys
import numpy as np

from CLI.utils import breaker

def main():
    args: tuple = ("--path", "-p")
    path: str = "data"
    if args in sys.argv: path = sys.argv[sys.argv.index(args) + 1]

    assert "images.npy" in os.listdir(path) and "labels.npy" in os.listdir(path), "Please run python np_make.py"

    images = np.load(os.path.join(path, "images.npy"))

    breaker()
    print("Mean\n")
    print(f"Red Channel Mean   : {images[:, :, 0].mean() / 255}")
    print(f"Green Channel Mean : {images[:, :, 1].mean() / 255}")
    print(f"Blue Channel Mean  : {images[:, :, 2].mean() / 255}")

    breaker()
    print("Standard Deviation\n")
    print(f"Red Channel Std   : {images[:, :, 0].std() / 255}")
    print(f"Green Channel Std : {images[:, :, 1].std() / 255}")
    print(f"Blue Channel Std  : {images[:, :, 2].std() / 255}")

    breaker()


if __name__ == "__main__":
    sys.exit(main() or 0)