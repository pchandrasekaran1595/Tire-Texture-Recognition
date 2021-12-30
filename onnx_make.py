import os
import sys
import torch

from CLI.utils import SAVE_PATH, DEVICE, breaker
from CLI.models import get_model

READ_PATH = "saves"
SAVE_PATH = "saves"


def main():
    args_1: tuple = ("--backbone", "-bb")
    args_2: str = "--size"
    args_3: str = "--opver"

    backbone = "mobilenet"
    size: int = 320
    opset_version: int = 9

    if args_1[0] in sys.argv: backbone = sys.argv[sys.argv.index(args_1[0]) + 1]
    if args_1[1] in sys.argv: backbone = sys.argv[sys.argv.index(args_1[1]) + 1]

    if args_2 in sys.argv: size = int(sys.argv[sys.argv.index(args_2) + 1])

    if args_3 in sys.argv: opset_version = int(sys.argv[sys.argv.index(args_3) + 1])

    filenames = os.listdir(READ_PATH)
    for name in filenames:
        if name[-3:] == ".pt":
            model_name = name
            break
    else:
        breaker()
        print("No Model File Detected")
        breaker()
    
    breaker()
    print("Building Model ...")
    model = get_model(0, "full", backbone)
    
    model.load_state_dict(torch.load(os.path.join(READ_PATH, model_name), map_location=DEVICE)["model_state_dict"])
    model.eval()
    model.to(DEVICE)

    batch_size = 1
    dummy = torch.randn(batch_size, 3, size, size).to(DEVICE)

    breaker()
    print("Exporting Model ...")
    torch.onnx.export(model=model, 
                      args=dummy, 
                      f=os.path.join(SAVE_PATH, "model.onnx"), 
                      input_names=["input"], 
                      output_names=["output"], 
                      opset_version=opset_version,
                      export_params=True,
                      training=torch.onnx.TrainingMode.EVAL,
                      dynamic_axes={
                        "input" : {0 : "batch_size"},
                        "output" : {0 : "batch_size"}
                        }
                    )
    
    breaker()

if __name__ == "__main__":
    sys.exit(main() or 0)
