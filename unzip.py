import os
import sys
import shutil
import zipfile

from CLI.utils import breaker


def unzip(path: str) -> None:
    with zipfile.ZipFile("data.zip", "r") as zip_ref:
        zip_ref.extractall(path)
        
    
def main():
    args: tuple = ("--path", "-p")
    path: str = "data"
    if args[0] in sys.argv: path = sys.argv[sys.argv.index(args[0]) + 1]
    if args[1] in sys.argv: path = sys.argv[sys.argv.index(args[1]) + 1]

    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)

    breaker()
    print("Unzipping Data ...")

    unzip(path)

    breaker()
    print("Unzipping Complete")
    breaker()


if __name__ == "__main__":
    sys.exit(main() or 0)