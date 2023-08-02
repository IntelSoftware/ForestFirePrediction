import torch
import intel_extension_for_pytorch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from model import FireFinder

device = "xpu" if torch.xpu.is_available() else "cpu"
print(f"using device: {device}")

# Image transformations
transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def save_model(model, path):
    model.to("cpu")
    torch.save(model.state_dict(), path)


def convert_save_torchscript(model, path):
    model.to("cpu")
    model = torch.jit.script(model)
    model.save(path)
    return model


def load_model(path):
    model = FireFinder()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def load_torchscript_model(path):
    model = torch.jit.load(path)
    model.eval()
    return model


def load_data(path):
    dataset = ImageFolder(path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    return dataloader


def run_inference(model, dataloader):
    results = []
    model = model.eval()
    model = model.to(device)
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            results.extend(preds.tolist())
    return results
