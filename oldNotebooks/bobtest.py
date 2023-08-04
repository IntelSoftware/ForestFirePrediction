import glob
from PIL import Image
from os.path import exists
import os
import torchvision.transforms.functional as TF
import intel_extension_for_pytorch as ipex
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import intel_extension_for_pytorch as ipex

batch_size = 128
model_name = "model_acc_94_device_xpu_lr_0.000214_epochs_12_jit"
TEST_DIR = 'data/shift/'

print(model_name)
model_read = torch.jit.load(f"models/{model_name}.pt")
model_read.eval()

def create_dataloader(
	directory: str, 
	batch_size: int, 
	shuffle: bool = False, 
	transform=None) -> DataLoader:
	data = datasets.ImageFolder(directory, transform=transform)
	return DataLoader(data, batch_size=batch_size, shuffle=shuffle)

test_dataloader = create_dataloader(
	TEST_DIR,
	batch_size=batch_size,
	shuffle=False,
	transform=None
)

transform = transforms.Compose([
	transforms.PILToTensor()])

for d,l in test_dataloader.dataset:
    img_tensor = transform(d)
    #img_XPU = img_tensor.to("xpu")
    score = model_read(img_tensor)
    print(score, l, d)
