import os
import pathlib
import warnings
import random
import time
import gc
from typing import Tuple
#import glob
from PIL import Image
from os.path import exists
import os
warnings.filterwarnings("ignore")

import torch
import intel_extension_for_pytorch
import numpy as np
import matplotlib.pyplot as plt
import wandb

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from batch_finder import optimum_batch_size
from config import set_seed, device
from data_loader import (
    TRAIN_DIR,
    VALID_DIR,
    augment_and_save,
    data_distribution,
    imagenet_stats,
    img_transforms,
    plot_data_distribution,
    show_data,
)
from metrics import Metrics
from model import FireFinder
from trainer import Trainer
from lr_finder import LearningRateFinder
from torch import optim

# hyper params
EPOCHS = 20 
DROPOUT = .6
# LR would be changed if we are using a LR finder
LR = 2.14e-4
#LR = 3.e-3
TEST_DIR = 'data/shift/'
BATCH_SIZE = 32 #128  # Default batch size

def create_dataloader(
    directory: str, batch_size: int, shuffle: bool = False, transform=None
) -> DataLoader:
    """
    Create a DataLoader from a directory of images.

    Args:
        directory (str): Directory containing images.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
        transform ([type], optional): Transformations to apply to the images. Defaults to None.

    Returns:
        DataLoader: DataLoader with images from the directory.
    """
    data = datasets.ImageFolder(directory, transform=transform)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)


def setup_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    """
    Setup train and validation DataLoaders.

    Args:
        config (dict): Configuration dictionary containing batch_size.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing train and validation dataloaders.
    """
    return create_dataloader(
        TRAIN_DIR, config["batch_size"], shuffle=True, transform=img_transforms["train"]
    ), create_dataloader(
        VALID_DIR, config["batch_size"], transform=img_transforms["valid"]
    )


def find_lr(model: FireFinder, optimizer: optim.Adam, dataloader: DataLoader) -> float:
    """
    Find best learning rate using Learning Rate Finder.

    Args:
        model (FireFinder): FireFinder model.
        optimizer (optim.Adam): Adam optimizer.
        dataloader (DataLoader): DataLoader with training data.

    Returns:
        float: Best learning rate.
    """
    lr_finder = LearningRateFinder(model, optimizer, device)
    best_lr = lr_finder.lr_range_test(dataloader, start_lr=1e-2, end_lr=1e-5)
    return best_lr


def train(model: FireFinder, trainer: Trainer, config: dict):
    """
    Train a FireFinder model.

    Args:
        model (FireFinder): FireFinder model.
        trainer (Trainer): Trainer to train the model.
        config (dict): Configuration dictionary containing learning rate and batch size.
    """
    train_dataloader, valid_dataloader = setup_dataloaders(config)
    print("training data")
    plot_data_distribution(data_distribution(train_dataloader.dataset, TRAIN_DIR))
    print("\nvalidation data")
    plot_data_distribution(data_distribution(valid_dataloader.dataset, VALID_DIR))
    print(f"______________")
    start = time.time()
    val_acc = trainer.fine_tune(train_dataloader, valid_dataloader)
    model_save_path = f"./models/model_acc_{val_acc}_device_{device}_lr_{trainer.lr}_epochs_{EPOCHS}.pt"
    torch.save(model.state_dict(), model_save_path)

    model.eval()
    with torch.no_grad():
        model_scripted = torch.jit.script(model)  # Export to TorchScript
        model_scripted.save(f"{model_save_path.replace('.pt','_jit.pt')}")  # Jit Save

    print(f"Model saved to :{model_save_path}")
    print(f"Time elapsed: {time.time() - start} seconds.")


def main(
    aug_data: bool = False,
    find_batch: bool = False,
    find_lr_rate: bool = False,
    use_wandb: bool = False,
    use_ipex=True,
):
    """
    Main function to execute the fine-tuning process.

    Args:
        aug_data (bool, optional): Whether to augment data. Defaults to False.
        find_batch (bool, optional): Whether to find optimal batch size. Defaults to False.
        find_lr_rate (bool, optional): Whether to find optimal learning rate. Defaults to False.
    """
    set_seed(42)
    print(f"Train folder {TRAIN_DIR}")
    print(f"Validation folder {VALID_DIR}")
    print(f"Using epoch: {EPOCHS}")
    print(f"Using Dropout: {DROPOUT}")
   
    batch_size = BATCH_SIZE

    if aug_data:
        print("Augmenting training and validation datasets...")
        t1 = time.time()
        augment_and_save(TRAIN_DIR)
        augment_and_save(VALID_DIR)
        print(f"Done Augmenting in {time.time() - t1} seconds...")

    model = FireFinder(simple=True, dropout=DROPOUT)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    if find_batch:
        print(f"Finding optimum batch size...")
        batch_size = optimum_batch_size(model, input_size=(3, 224, 224))
    print(f"Using batch size: {batch_size}")

    best_lr = LR
    if find_lr_rate:
        print("Finding best init lr...")
        train_dataloader = create_dataloader(
            TRAIN_DIR,
            batch_size=batch_size,
            shuffle=True,
            transform=img_transforms["train"],
        )
        best_lr = find_lr(model, optimizer, train_dataloader)
        del model, optimizer
        gc.collect()
        if device == torch.device("xpu"):
            torch.xpu.empty_cache()
    print(f"Using learning rate: {best_lr}")

    model = FireFinder(simple=True, dropout=DROPOUT)
    trainer = Trainer(
        model=model,
        optimizer=optim.Adam,
        lr=best_lr,
        epochs=EPOCHS,
        device=device,
        use_wandb=use_wandb,
        use_ipex=use_ipex,
    )
    train(model, trainer, config={"lr": best_lr, "batch_size": batch_size})

if __name__ == "__main__":
    main(
        aug_data=False, find_batch=False, find_lr_rate=False, use_wandb=True, use_ipex=False
    )
