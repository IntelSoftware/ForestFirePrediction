import warnings

warnings.filterwarnings("ignore")

import torch
import intel_extension_for_pytorch

from torchvision.models import resnet18 as resnet18_model
from torchvision.models.resnet import ResNet18_Weights
from diffusers import StableDiffusionImg2ImgPipeline


def download_and_cache_resnet18():
    print("Downloading and caching ResNet-18 model...")
    model = resnet18_model(pretrained=ResNet18_Weights.IMAGENET1K_V1)
    print("ResNet-18 model downloaded and cached successfully!")


def download_and_cache_stable_diffusion_pipeline(torch_dtype):
    print("Downloading and caching Stable Diffusion pipeline model...")
    model_id = "runwayml/stable-diffusion-v1-5"
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, torch_dtype=torch_dtype
    )
    print("Stable Diffusion pipeline model downloaded and cached successfully!")


if __name__ == "__main__":
    download_and_cache_resnet18()
    download_and_cache_stable_diffusion_pipeline(torch.float16)
