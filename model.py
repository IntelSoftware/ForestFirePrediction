from config import torch
import torch.nn as nn
import torchvision.models as models


class FireFinder(nn.Module):
    """
    A model to classify aerial images that could potentially Fire from satellite images
    We are using a pretrained resnet backbone model
    and images given to model are classified into one of 3 classes.
    0 - no Fire
    1 - Fire

    We currently use the resnet50 model as a backbone
    """

    def __init__(
        self,
        backbone="resnet18",
        simple=True,
        dropout=0.4,
        n_classes=2,
        feature_extractor=False,
    ):
        super(FireFinder, self).__init__()
        backbones = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "efficientnet_b0": lambda pretrained: models.efficientnet_b0(
                pretrained=pretrained
            ),
        }
        try:
            self.network = backbones[backbone](pretrained=True)
            if backbone == "efficientnet_b0":
                self.network.classifier[1] = nn.Linear(1280, n_classes)
            else:
                self.network.fc = nn.Linear(self.network.fc.in_features, n_classes)
        except KeyError:
            raise ValueError(f"Backbone model '{backbone}' not found")

        if feature_extractor:
            print("Running in future extractor mode.")
            for param in self.network.parameters():
                param.requires_grad = False
        else:
            print("Running in Finetuning mode.")

        for m, p in zip(self.network.modules(), self.network.parameters()):
            if isinstance(m, nn.BatchNorm2d):
                p.requires_grad = False

        if not simple and backbone != "efficientnet_b0":
            fc = nn.Sequential(
                nn.Linear(self.network.fc.in_features, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, n_classes),
            )
            for layer in fc.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
            self.network.fc = fc

    def forward(self, x_batch):
        return self.network(x_batch)
