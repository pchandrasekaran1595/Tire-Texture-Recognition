
import os
import re
import cv2
import json
import torch
import imgaug
import numpy as np
import matplotlib.pyplot as plt

from time import time
from imgaug import augmenters
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as DL
from sklearn.model_selection import KFold

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM_FINAL = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.41488, 0.41446, 0.41412], [0.20391, 0.20348, 0.20304])])

SAVE_PATH = "saves"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


class DS(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray = None, transform=None, mode: str = "train"):

        assert re.match(r"^train$", mode, re.IGNORECASE) or re.match(r"^valid$", mode, re.IGNORECASE) or re.match(r"^test$", mode, re.IGNORECASE), "Invalid Mode"
        
        self.mode = mode
        self.transform = transform
        self.images = images

        if re.match(r"^train$", mode, re.IGNORECASE) or re.match(r"^valid$", mode, re.IGNORECASE):
            self.labels = labels

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        if re.match(r"^train$", self.mode, re.IGNORECASE) or re.match(r"^valid$", self.mode, re.IGNORECASE):
            return self.transform(self.images[idx]), torch.FloatTensor(self.labels[idx])
        else:
            return self.transform(self.images[idx])


def breaker(num: int = 50, char: str = "*") -> None:
    print("\n" + num*char + "\n")


def load_data(path: str) -> tuple:
    assert "images.npy" in os.listdir(path) and "labels.npy" in os.listdir(path), "Please run python np_make.py"

    images = np.load(os.path.join(path, "images.npy"))
    labels = np.load(os.path.join(path, "labels.npy"))

    return images, labels


def get_augment(seed: int):
    imgaug.seed(seed)
    augment = augmenters.Sequential([
        augmenters.HorizontalFLip(p=0.15),
        augmenters.VerticalFLip(p=0.15),
        augmenters.Affine(scale=(0.5, 1.5), translate_percent=(-0.1, 0.1), rotate=(-45, 45)),
    ])
    return augment


def prepare_train_and_valid_dataloaders(path: str, mode: str, batch_size: int, seed: int, augment: bool=False):

    images, labels = load_data(path)

    for tr_idx, va_idx in KFold(n_splits=5, shuffle=True, random_state=seed).split(images):
        tr_images, va_images, tr_labels, va_labels = images[tr_idx], images[va_idx], labels[tr_idx], labels[va_idx]
        break

    if augment:
        augmenter = get_augment(seed)
        tr_images = augmenter(images=tr_images)
    
    if re.match(r"^full$", mode, re.IGNORECASE) or re.match(r"^semi$", mode, re.IGNORECASE):
        tr_data_setup = DS(tr_images, tr_labels, TRANSFORM, "train")
        va_data_setup = DS(va_images, va_labels, TRANSFORM, "valid")
    else:
        tr_data_setup = DS(tr_images, tr_labels, TRANSFORM_FINAL, "train")
        va_data_setup = DS(va_images, va_labels, TRANSFORM_FINAL, "valid")

    dataloaders = {
        "train" : DL(tr_data_setup, batch_size=batch_size, shuffle=True, generator=torch.manual_seed(seed)),
        "valid" : DL(va_data_setup, batch_size=batch_size, shuffle=False)
    }

    return dataloaders


def save_graphs(L: list, A: list) -> None:
    TL, VL, TA, VA = [], [], [], []
    for i in range(len(L)):
        TL.append(L[i]["train"])
        VL.append(L[i]["valid"])
        TA.append(A[i]["train"])
        VA.append(A[i]["valid"])
    x_Axis = np.arange(1, len(TL) + 1)
    plt.figure("Plots")
    plt.subplot(1, 2, 1)
    plt.plot(x_Axis, TL, "r", label="Train")
    plt.plot(x_Axis, VL, "b", label="Valid")
    plt.legend()
    plt.grid()
    plt.title("Loss Graph")
    plt.subplot(1, 2, 2)
    plt.plot(x_Axis, TA, "r", label="Train")
    plt.plot(x_Axis, VA, "b", label="Valid")
    plt.legend()
    plt.grid()
    plt.title("Accuracy Graph")
    plt.savefig(os.path.join(SAVE_PATH, "Graphs.jpg"))
    plt.close("Plots")
    

def fit(model=None, optimizer=None, scheduler=None, epochs=None, early_stopping_patience=None, 
        dataloaders=None, verbose=False):
    
    def get_accuracy(y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)

        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0

        return torch.count_nonzero(y_pred == y_true).item() / len(y_pred)
    
    if verbose:
        breaker()
        print("Training ...")
        breaker()

    bestLoss, bestAccs = {"train" : np.inf, "valid" : np.inf}, {"train" : 0.0, "valid" : 0.0}
    Losses, Accuracies = [], []
    name = "state.pt"

    start_time = time()
    for e in range(epochs):
        e_st = time()
        epochLoss, epochAccs = {"train" : 0.0, "valid" : 0.0}, {"train" : 0.0, "valid" : 0.0}

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            
            lossPerPass, accsPerPass = [], []

            for X,y in dataloaders[phase]:
                X, y = X.to(DEVICE), y.to(DEVICE)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    output = model(X)
                    loss = torch.nn.BCEWithLogitsLoss()(output, y)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                lossPerPass.append(loss.item())
                accsPerPass.append(get_accuracy(output, y))
            epochLoss[phase] = np.mean(np.array(lossPerPass))
            epochAccs[phase] = np.mean(np.array(accsPerPass))
        Losses.append(epochLoss)
        Accuracies.append(epochAccs)

        if early_stopping_patience:
            if epochLoss["valid"] < bestLoss["valid"]:
                bestLoss = epochLoss
                BLE = e + 1
                torch.save({"model_state_dict": model.state_dict(),
                            "optim_state_dict": optimizer.state_dict()},
                           os.path.join(SAVE_PATH, name))
                early_stopping_step = 0
            else:
                early_stopping_step += 1
                if early_stopping_step > early_stopping_patience:
                    print("\nEarly Stopping at Epoch {}".format(e + 1))
                    break
        
        if epochLoss["valid"] < bestLoss["valid"]:
            bestLoss = epochLoss
            BLE = e + 1
            torch.save({"model_state_dict" : model.state_dict(),
                        "optim_state_dict" : optimizer.state_dict()},
                        os.path.join(SAVE_PATH, name))
        
        if epochAccs["valid"] > bestAccs["valid"]:
            bestAccs = epochAccs
            BAE = e + 1
        
        if scheduler:
            scheduler.step(epochLoss["valid"])
        
        if verbose:
            print("Epoch: {} | Train Loss: {:.5f} | Valid Loss: {:.5f} |\
Train Accs: {:.5f} | Valid Accs: {:.5f} | Time: {:.2f} seconds".format(e+1, 
                                                                       epochLoss["train"], epochLoss["valid"], 
                                                                       epochAccs["train"], epochAccs["valid"], 
                                                                       time()-e_st))

    if verbose:                                           
        breaker()
        print(f"Best Validation Loss at Epoch {BLE}")
        breaker()
        print(f"Best Validation Accs at Epoch {BAE}")
        breaker()
        print("Time Taken [{} Epochs] : {:.2f} minutes".format(len(Losses), (time()-start_time)/60))
        breaker()
        print("Training Completed")
        breaker()

    return Losses, Accuracies, BLE, BAE, name


def predict(model=None, mode: str = None, image_path: str = None, size: int = 320) -> str:
    model.load_state_dict(torch.load(os.path.join(SAVE_PATH, "state.pt"), map_location=DEVICE)["model_state_dict"])
    model.eval()
    model.to(DEVICE)

    image = cv2.resize(src=cv2.cvtColor(src=cv2.imread(image_path, cv2.IMREAD_COLOR), code=cv2.COLOR_BGR2RGB), dsize=(size, size), interpolation=cv2.INTER_AREA)
    labels = json.load(open("labels.json", "r"))

    with torch.no_grad():
        if re.match(r"^full$", mode, re.IGNORECASE) or re.match(r"^semi$", mode, re.IGNORECASE):
            output = torch.sigmoid(model(TRANSFORM(image).to(DEVICE).unsqueeze(dim=0)))
        else:
            output = torch.sigmoid(model(TRANSFORM_FINAL(image).to(DEVICE).unsqueeze(dim=0)))
        
    if output.item() > 0.5:
        output = 1
    else:
        output = 0
    
    return labels[str(output)].title()
