import os
import sys
import cv2
import numpy as np

from CLI.utils import breaker


def reserve_memory(path: str, size: int) -> tuple:
    total_num_files: int = 0
    for name_1 in os.listdir(path):
        if name_1 == "test" or name_1 == "testing_data" or name_1[-3:] == "npy":
            pass
        else:
            for name_2 in os.listdir(os.path.join(path, name_1)):
                total_num_files += len(os.listdir(os.path.join(os.path.join(path, name_1), name_2)))
    images = np.zeros((total_num_files, size, size, 3), dtype=np.uint8)
    labels = np.ones((total_num_files, 1), dtype=np.uint8)
    return images, labels


def main() -> None:
    args_1: tuple = ("--path", "-p")
    args_2: tuple = ("--size", "-s")
    args_3: tuple = ("--kaggle", "-k")

    path: str = "data"
    size: int = 320
    kaggle: bool = False

    if args_1[0] in sys.argv: path = sys.argv[sys.argv.index(args_1[0]) + 1]
    if args_1[1] in sys.argv: path = sys.argv[sys.argv.index(args_1[1]) + 1]

    if args_2[0] in sys.argv: size = int(sys.argv[sys.argv.index(args_2[0]) + 1])
    if args_2[1] in sys.argv: size = int(sys.argv[sys.argv.index(args_2[1]) + 1])

    if args_3[0] in sys.argv or args_3[1] in sys.argv: kaggle = True

    assert os.path.exists(path), "Please Unzip the data first using python unzip.py"

    breaker()
    print("Reserving Memory ...")

    images, labels = reserve_memory(path, size)

    i: int = 0
    j: int = 0

    breaker()
    print("Saving Images to Numpy Arrays ...")

    folders = sorted([folder_name for folder_name in os.listdir(path) if folder_name != "test" and folder_name != "testing_data" and folder_name[-3:] != "npy"])

    for folder_name_1 in folders:
        for folder_name_2 in sorted(os.listdir(os.path.join(path, folder_name_1))):
            for filename in os.listdir(os.path.join(os.path.join(path, folder_name_1), folder_name_2)):
                images[i] = cv2.resize(src=cv2.cvtColor(src=cv2.imread(os.path.join(os.path.join(os.path.join(path, folder_name_1), folder_name_2), filename), cv2.IMREAD_COLOR), code=cv2.COLOR_BGR2RGB), dsize=(size, size), interpolation=cv2.INTER_AREA)
                labels[i] = labels[i] * j
                i += 1
            j += 1

    if kaggle:
        np.save("./images.npy", images)
        np.save("./labels.npy", labels)
    else:
        np.save(os.path.join(path, "images.npy"), images)
        np.save(os.path.join(path, "labels.npy"), labels)

    breaker()
    print("Saving Complete")
    breaker()


if __name__ == "__main__":
    sys.exit(main() or 0)