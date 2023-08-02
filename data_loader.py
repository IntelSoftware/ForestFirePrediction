from config import torch
from config import device
from config import seed_everything

import pathlib
import plotext as tplt

from torchvision import transforms
import matplotlib.pyplot as plt
import fnmatch
import os

import cv2
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Compose

imagenet_stats = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
data_dir = pathlib.Path("./data/shift/")
TRAIN_DIR = data_dir / "train"
VALID_DIR = data_dir / "val"


img_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ToTensor(),
            transforms.Normalize(*imagenet_stats),
        ]
    ),
    "valid": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(*imagenet_stats),
        ]
    ),
}


def augment_and_save(path, target_number=1000):
    """augment dataset if total number per class is less than 1000 and save to data dir."""
    subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    for subfolder in subfolders:
        images = fnmatch.filter(os.listdir(subfolder), "*.png")
        augmentations_per_image = max(target_number // len(images), 1)
        augmentations = Compose(
            [
                HorizontalFlip(),
                VerticalFlip(),
                RandomRotate90(),
            ]
        )
        for image in images:
            image_path = os.path.join(subfolder, image)
            img = cv2.imread(image_path)
            for i in range(augmentations_per_image):
                augmented = augmentations(image=img)
                new_filename = os.path.splitext(image)[0] + f"_{i}.png"
                cv2.imwrite(
                    os.path.join(subfolder, new_filename),
                    augmented["image"],
                )


def _denormalize(images, imagenet_stats):
    """De-normalize dataset using imagenet std and mean to show images."""
    mean = torch.tensor(imagenet_stats[0]).reshape(1, 3, 1, 1)
    std = torch.tensor(imagenet_stats[1]).reshape(1, 3, 1, 1)
    return images * std + mean


def show_data(dataloader, imagenet_stats=imagenet_stats, num_data=2):
    """Show `num_data` of images and labels from dataloader."""
    batch = next(iter(dataloader))
    imgs, labels = batch[0][:num_data].to(device), batch[1][:num_data].tolist()

    if plt.get_backend() == "agg":
        print(f"Labels for {num_data} images: {labels}")
    else:
        _, axes = plt.subplots(1, num_data, figsize=(10, 6))
        for n in range(num_data):
            axes[n].set_title(labels[n])
            imgs[n] = _denormalize(imgs[n], imagenet_stats)
            axes[n].imshow(torch.clamp(imgs[n].cpu(), 0, 1).permute(1, 2, 0))
        plt.show()


def data_distribution(dataset, path: str) -> dict:
    """
    Returns a dictionary with the distribution of each class in the dataset.
    """
    class_counts = {
        cls: len(fnmatch.filter(os.listdir(f"{path}/{cls}"), "*.png"))
        for cls in dataset.class_to_idx.keys()
    }
    return class_counts


def plot_data_distribution(data_dist: dict, title: str = ""):
    import seaborn as sns
    classes, counts = list(data_dist.keys()), list(data_dist.values())

    if plt.get_backend() == "agg":
        tplt.simple_bar(classes, counts, width=100, title=title)
        tplt.show()
    else:
        sns.barplot(x=classes, y=counts).set_title(title)
        plt.show()
