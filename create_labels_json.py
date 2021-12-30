import sys
import json


def main():

    labels = {
        "0"  : "Cracked",
        "1"  : "Normal",
    }

    json.dump(labels, open("labels.json", "w"))


if __name__ == "__main__":
    sys.exit(main() or 0)