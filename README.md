# Forest Fire Prediction

### Learning Objectives:

- Apply Intel Extension for PyTorch to convert model and data to xpu 
- Apply fine tuning to satellite images to predict fire fires 2 years in advance
- Generate syntheic satellite images using Stable Diffusion
- Optimize a 4 part Stable Diffusion pipeline using IPEX

## Syllabus:



| Modules | Description | Duration |
| :----- | :------ | :------ |
|[00_LocalSetup](00_LocalSetup.ipynb) | Setup seesion on IDC | 20 minutes |
|[01_Overview_Finetuning.ipynb](01_Overview_Finetuning.ipynb)| + Describe overview of Forest Fire Prediction Workshop<br>+ Describe background and impact of forest fires on humanity<br>+ Provide overview of the model flow<br>+ Describe the datasets used (NASA/MODIS, USDA/NAIP).<br>+ Demonstrate how to label data.| 15 min |
|[02_Prepare_Data.ipynb](02_Prepare_Data.ipynb)| + Describe how to prepare & label data.| skip |
|[03_Finetuning.ipynb](03_Finetuning.ipynb)| + Describe overview of fine tuning<br>+ Describe required folder structure for training<br>+ Describe how to implement Intel(r) Extension for PyTorch (IPEX).<br>+ Apply changes to target xpu and accelerate model with IPEX.<br>+ Apply conversion of models and data to xpu optimized format.<br>+ Convert model back to CPU format for saving.<br>| 30 min |
|[04_ScoreMatrix.ipynb](04_ScoreMatrix.ipynb)| + Describe the confusions matrix and overall suitability of model to predicting forest fires two years in advance.<br>| 10 min |
|[05_StableDiffusionAccelerated.ipynb](05_StableDiffusionAccelerated.ipynb)| + Describe the the Stable Diffusion model and what it is used for.<br>+ Apply changes to text to generate new synthetic images derived from a handful of real images.<br>+ Compare times of IPEX accelerated xpu runs to Non Accelerated xpu runs| 30 min |
|[06_AcquiringAerialPhotos.ipynb](06_AcquiringAerialPhotos.ipynb)| + Describe three sources for acquiring real aerial photos.<br>+ Describe Javascript codes required for Google Earth Engine approach.<br>+ Describe search and selection approach using USGA Earth Explorer.<br>+ Describe pros and cons of the various sources| 10 min |

# FAQ

- Q: What is the minimum acreage MODIS considers a burn area? A: Not spcified
- Q: Define a burn area - Where a fire starts is not the same than how big the fire got.  Maybe  many small fires contribued to a burn area - What is the MODIS definition of burn area? A: Not specified
- Q: Have you considered instead of using USDA, to feed the model with land use map, orientation map, elevation map and slope map or other sources even tabular sources? A: a multi modal approach such as this was considered but for sake of time and complexity I gave up on the effort.
- Q: How will the current CA and NV fire right now would influence the relevance of your notebook? A: the MODIS dataset does not yest have information about current burn areas. However a model trained on images from surrounding areas and times woudl likely be predictive if pre-fire iamges had been fed to a well trained model
- Q: What is the impact of forest fires, how many lives and how much money the fires cost in CA? A: this is addressed in the [01_Overview_Finetuning.ipynb](01_Overview_Finetuning.ipynb) module.
- Q: What hardware does this workshop give us access to? A: You will be doing experiments on the Intel(r) 4th Generation Scalable Xeon processors equipped with Intel(r) Data Center GPU max series accelerators (xpu)