import os
import re
import sys
import cv2
import onnx
import json
import numpy as np
import onnxruntime as ort

from CLI.utils import breaker

READ_PATH = "saves"


def get_image(path: str, size: int, mean: list, std: list) -> np.ndarray:
    image = cv2.resize(src=cv2.cvtColor(src=cv2.imread(path, cv2.IMREAD_COLOR), code=cv2.COLOR_BGR2RGB), dsize=(size, size), interpolation=cv2.INTER_AREA)
    image = image / 255
    for i in range(image.shape[2]):
        image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    return image.astype("float32")


def infer(image: np.ndarray) -> str:
    model_path = os.path.join(READ_PATH, "model.onnx")
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    ort_session = ort.InferenceSession(model_path)

    output = ort_session.run(None, {"input" : image.astype("float32")})
    labels = json.load(open("labels.json", "r"))

    return labels[str(np.argmax(output[0]))].title()


def main():
    assert "model.onnx" in os.listdir(READ_PATH), "Please run make_onnx.py first"

    args_1: tuple = ("--file", "-f")
    args_2: tuple = ("--mode", "-m")
    args_3: str = "--size"

    name: str = None
    mode: str = "full"
    size: int = 320
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if args_1[0] in sys.argv: name = sys.argv[sys.argv.index(args_1[0]) + 1]
    if args_1[1] in sys.argv: name = sys.argv[sys.argv.index(args_1[1]) + 1]

    if args_3 in sys.argv: size = int(sys.argv[sys.argv.index(args_3) + 1])

    assert name is not None, "Enter a filename using --file | -f"

    if re.match(r"^full$", mode, re.IGNORECASE) or re.match(r"^semi$", mode, re.IGNORECASE):
        mean, std = [0.41488, 0.41446, 0.41412], [0.20391, 0.20348, 0.20304]
    
    image = get_image("data/test/" + name, size, mean, std)

    breaker()
    print(f"Label : {infer(image)}")
    breaker()


if __name__ == "__main__":
    sys.exit(main() or 0)
