import re
import torch
from torch import nn, optim
from torchvision import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, mode: str, model_name: str):
        super(Model, self).__init__()

        self.mode = mode
        self.model_name = model_name

        if re.match(r"^vgg$", self.model_name, re.IGNORECASE):
            if re.match(r"^full$", self.mode, re.IGNORECASE):
                self.model = models.vgg16_bn(pretrained=False, progress=True)
                self.model.classifier[-1] = nn.Linear(in_features=self.model.classifier[-1].in_features, out_features=1)
            elif re.match(r"^semi$", self.mode, re.IGNORECASE) or re.match(r"^final$", self.mode, re.IGNORECASE):
                self.model = models.vgg16_bn(pretrained=True, progress=True)
                self.freeze()
                self.model.classifier[-1] = nn.Linear(in_features=self.model.classifier[-1].in_features, out_features=1)
            
        elif re.match(r"^resnet$", self.model_name, re.IGNORECASE):
            if re.match(r"^full$", self.mode, re.IGNORECASE):
                self.model = models.resnet50(pretrained=False, progress=True)
                self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=1)
            elif re.match(r"^semi$", self.mode, re.IGNORECASE) or re.match(r"^final$", self.mode, re.IGNORECASE):
                self.model = models.resnet50(pretrained=True, progress=True)
                self.freeze()
                self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=1)
        
        elif re.match(r"^densenet$", self.model_name, re.IGNORECASE):
            if re.match(r"^full$", self.mode, re.IGNORECASE):
                self.model = models.densenet169(pretrained=False, progress=True)
                self.model.classifier = nn.Linear(in_features=self.model.classifier.in_features, out_features=1)
            elif re.match(r"^semi$", self.mode, re.IGNORECASE) or re.match(r"^final$", self.mode, re.IGNORECASE):
                self.model = models.densenet169(pretrained=False, progress=True)
                self.freeze()
                self.model.classifier = nn.Linear(in_features=self.model.classifier.in_features, out_features=1)
        
        elif re.match(r"^mobilenet$", self.model_name, re.IGNORECASE):
            if re.match(r"^full$", self.mode, re.IGNORECASE):
                self.model = models.mobilenet_v3_small(pretrained=False, progress=True)
                self.model.classifier[-1] = nn.Linear(in_features=self.model.classifier[-1].in_features, out_features=1)
            elif re.match(r"^semi$", self.mode, re.IGNORECASE) or re.match(r"^final$", self.mode, re.IGNORECASE):
                self.model = models.mobilenet_v3_small(pretrained=False, progress=True)
                self.freeze()
                self.model.classifier[-1] = nn.Linear(in_features=self.model.classifier[-1].in_features, out_features=1)

    def freeze(self):
        for params in self.parameters():
            params.requires_grad = False

        if re.match(r"^vgg$", self.model_name, re.IGNORECASE):
            if re.match(r"^semi$", self.mode, re.IGNORECASE):
                for names, params in self.named_parameters():
                    if re.match(r".*features.3[4-9].*", names, re.IGNORECASE) or re.match(r".*features.4[0-9].*", names, re.IGNORECASE) or re.match(r".*classifier.*", names, re.IGNORECASE):
                        params.requires_grad = True
        
        elif re.match(r"^resnet$", self.model_name, re.IGNORECASE):
            if re.match(r"^semi$", self.mode, re.IGNORECASE):
                for names, params in self.named_parameters():
                    if re.match(r".*layer4.*", names, re.IGNORECASE):
                        params.requires_grad = True
        
        elif re.match(r"^densenet$", self.model_name, re.IGNORECASE):
            if re.match(r"^semi$", self.mode, re.IGNORECASE):
                for names, params in self.named_parameters():
                    if re.match(r".*denseblock4.*", names, re.IGNORECASE) or re.match(r".*norm5.*", names, re.IGNORECASE):
                        params.requires_grad = True
        
        elif re.match(r"^mobilenet$", self.model_name, re.IGNORECASE):
            if re.match(r"^semi$", self.mode, re.IGNORECASE):
                for names, params in self.named_parameters():
                    if re.match(r".*features.9.*", names, re.IGNORECASE) or re.match(r".*features.1[0-2].*", names, re.IGNORECASE) or re.match(r".*classifier.*", names, re.IGNORECASE):
                        params.requires_grad = True

    def get_optimizer(self, lr: float = 1e-3, wd: float = 0.0):
        params = [p for p in self.parameters() if p.requires_grad]
        return optim.Adam(params, lr=lr, weight_decay=wd)
    
    def get_plateau_scheduler(self, optimizer=None, patience: int = 5, eps: float = 1e-8):
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=patience, eps=eps, verbose=True)

    def forward(self, x):
        return nn.LogSoftmax(dim=1)(self.model(x))


def get_model(seed: int, mode: str, model_name: str):
    torch.manual_seed(seed)
    model = Model(mode, model_name).to(DEVICE)

    return model
