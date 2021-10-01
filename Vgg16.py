import torch
from torchvision import models


def net():
    model = models.vgg16(pretrained=True)
    # print("Vgg16:\n", model)
    for parma in model.parameters():
        parma.requires_grad = False
    #
    model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(p=0.5),
                                           torch.nn.Linear(4096, 4096),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(p=0.5),
                                           torch.nn.Linear(4096, 1))
    return model
