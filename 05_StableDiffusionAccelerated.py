import os
import warnings

warnings.filterwarnings("ignore")

import random
import requests
import torch
import intel_extension_for_pytorch as ipex
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline
import torch.nn as nn
import time
from typing import List, Dict, Tuple


class Img2ImgModel:
    """
    This class creates a model for transforming images based on given prompts.
    """

    def __init__(
        self,
        model_id_or_path: str,
        device: str = "xpu",
        torch_dtype: torch.dtype = torch.float16,
        optimize: bool = True,
    ) -> None:
        """
        Initialize the model with the specified parameters.

        Args:
            model_id_or_path (str): The ID or path of the pre-trained model.
            device (str, optional): The device to run the model on. Defaults to "xpu".
            torch_dtype (torch.dtype, optional): The data type to use for the model. Defaults to torch.float16.
            optimize (bool, optional): Whether to optimize the model. Defaults to True.
        """
        self.device = device
        self.pipeline = self._load_pipeline(model_id_or_path, torch_dtype)
        if optimize:
            start_time = time.time()
            print("Optimizing the model...")
            self.optimize_pipeline()
            print(
                "Optimization completed in {:.2f} seconds.".format(
                    time.time() - start_time
                )
            )

    def _load_pipeline(
        self, model_id_or_path: str, torch_dtype: torch.dtype
    ) -> StableDiffusionImg2ImgPipeline:
        """
        Load the pipeline for the model.

        Args:
            model_id_or_path (str): The ID or path of the pre-trained model.
            torch_dtype (torch.dtype): The data type to use for the model.

        Returns:
            StableDiffusionImg2ImgPipeline: The loaded pipeline.
        """
        print("Loading the model...")
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id_or_path, torch_dtype=torch_dtype
        )
        pipeline = pipeline.to(self.device)
        print("Model loaded.")
        return pipeline

    def _optimize_pipeline(
        self, pipeline: StableDiffusionImg2ImgPipeline
    ) -> StableDiffusionImg2ImgPipeline:
        """
        Optimize the pipeline of the model.

        Args:
            pipeline (StableDiffusionImg2ImgPipeline): The pipeline to optimize.

        Returns:
            StableDiffusionImg2ImgPipeline: The optimized pipeline.
        """
        for attr in dir(pipeline):
            if isinstance(getattr(pipeline, attr), nn.Module):
                setattr(
                    pipeline,
                    attr,
                    ipex.optimize(
                        getattr(pipeline, attr).eval(),
                        dtype=pipeline.text_encoder.dtype,
                        inplace=True,
                    ),
                )
        return pipeline

    def optimize_pipeline(self) -> None:
        """
        Optimize the pipeline of the model.
        """
        self.pipeline = self._optimize_pipeline(self.pipeline)

    def get_image_from_url(self, url: str, path: str) -> Image.Image:
        """
        Get an image from a URL or from a local path if it exists.

        Args:
            url (str): The URL of the image.
            path (str): The local path of the image.

        Returns:
            Image.Image: The loaded image.
        """
        if os.path.exists(path):
            img = Image.open(path).convert("RGB")
        else:
            response = requests.get(url)
            if response.status_code != 200:
                raise Exception(
                    f"Failed to download image. Status code: {response.status_code}"
                )
            if not response.headers["content-type"].startswith("image"):
                raise Exception(
                    f"URL does not point to an image. Content type: {response.headers['content-type']}"
                )
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img.save(path)
        img = img.resize((768, 512))
        return img

    @staticmethod
    def random_sublist(lst):
        sublist = []
        for _ in range(random.randint(1, len(lst))):
            item = random.choice(lst)
            sublist.append(item)
        return sublist

    def generate_images(
        self,
        prompt: str,
        image_url: str,
        class_name: str,
        seed_image_identifier: str,
        variations: List[str],
        num_images: int = 5,
        strength: float = 0.75,
        guidance_scale: float = 7.5,
        save_path: str = "output",
        seed_path: str = "intput",
    ) -> List[Image.Image]:
        """
        Generate images based on the provided prompt and variations.

        Args:
            prompt (str): The base prompt for the generation.
            image_url (str): The URL of the seed image.
            class_name (str): The class of the image (e.g. "fire" or "no_fire").
            seed_image_identifier (str): The identifier of the seed image.
            variations (List[str]): The list of variations to apply to the prompt.
            num_images (int, optional): The number of images to generate. Defaults to 5.
            strength (float, optional): The strength of the transformation. Defaults to 0.75.
            guidance_scale (float, optional): The scale of the guidance. Defaults to 7.5.
            save_path (str, optional): The path to save the generated images. Defaults to "output".
            seed_path (str, optional): The path to save the input images. Defaults to "input".

        Returns:
            List[Image.Image]: The list of generated images.
        """
        input_image_path = f"{seed_path}/{seed_image_identifier}.png"
        init_image = self.get_image_from_url(image_url, input_image_path)
        images = []
        for i in range(num_images):
            variation = variations[i % len(variations)]
            final_prompt = f"{prompt} {variation}"
            image = self.pipeline(
                prompt=final_prompt,
                image=init_image,
                strength=strength,
                guidance_scale=guidance_scale,
            ).images
            output_image_path = os.path.join(
                save_path,
                f"{seed_image_identifier}_{'_'.join(variation.split())}_{i}.png",
            )
            image[0].save(output_image_path)
            images.append(image)
        return images


if __name__ == "__main__":
    model_id = "runwayml/stable-diffusion-v1-5"
    base_prompt = "A close image to this original satellite image with slight change in location"
    fire_variations = [
        "early morning with a wild fire",
        "late afternoon",
        "mid-day",
        "night with wild fire",
        "smoky conditions",
        "visible fire lines",
    ]
    no_fire_variations = [
        "early morning with clear skies",
        "no signs of fire",
        "night",
        "late afternoon with clear skies",
        "mid-day with clear skies",
        "with dense vegetation",
        "with sparse vegetation",
    ]

    image_urls = {
        "fire": [
            "https://github.com/rahulunair/ForestFirePrediction/blob/main/data/output/train/Fire/m_3912105_sw_10_h_20160713.png?raw=true",
            "https://github.com/rahulunair/ForestFirePrediction/blob/main/data/output/train/Fire/m_3912113_sw_10_h_20160713.png?raw=true",
            "https://github.com/rahulunair/ForestFirePrediction/blob/main/data/output/train/Fire/m_3912114_se_10_h_20160806.png?raw=true",
            "https://github.com/rahulunair/ForestFirePrediction/blob/main/data/output/train/Fire/m_3912120_ne_10_h_20160713.png?raw=true",
            "https://github.com/rahulunair/ForestFirePrediction/blob/main/data/output/train/Fire/m_4012355_se_10_h_20160713.png?raw=true",
        ],
        "no_fire": [
            "https://github.com/rahulunair/ForestFirePrediction/blob/main/data/output/train/NoFire/m_3912045_ne_10_h_20160712.png?raw=true",
            "https://github.com/rahulunair/ForestFirePrediction/blob/main/data/output/train/NoFire/m_3912057_sw_10_h_20160711.png?raw=true",
            "https://github.com/rahulunair/ForestFirePrediction/blob/main/data/output/train/NoFire/m_3912142_sw_10_h_20160711.png?raw=true",
            "https://github.com/rahulunair/ForestFirePrediction/blob/main/data/output/train/NoFire/m_3912343_se_10_h_20160529.png?raw=true",
            "https://github.com/rahulunair/ForestFirePrediction/blob/main/data/output/train/NoFire/m_4012241_se_10_h_20160712.png?raw=true",
        ],
    }

    model = Img2ImgModel(model_id, device="xpu")
    num_images = 5
    gen_img_count = 0

    try:
        start_time = time.time()
        for class_name, urls in image_urls.items():
            for url in urls:
                seed_image_identifier = os.path.basename(url).split(".")[0]
                input_dir = f"./input/{class_name}"
                output_dir = f"./output/{class_name}"
                os.makedirs(input_dir, exist_ok=True)
                os.makedirs(output_dir, exist_ok=True)
                variations = (
                    fire_variations if class_name == "fire" else no_fire_variations
                )
                model.generate_images(
                    base_prompt,
                    url,
                    class_name,
                    seed_image_identifier,
                    variations=variations,
                    save_path=output_dir,
                    seed_path=input_dir,
                    num_images=num_images,
                )
                gen_img_count += num_images
    except KeyboardInterrupt:
        print("\nUser interrupted image generation...")
    finally:
        print(
            f"Complete generating {gen_img_count} images in {'/'.join(output_dir.split('/')[:-1])} in {time.time() - start_time:.2f} seconds."
        )
