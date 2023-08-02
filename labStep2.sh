conda create -y -n PT2 intelpython3_full
conda activate PT2
conda install -y -c conda-forge jupyterlab
conda install -y -c anaconda seaborn pandas tqdm
pip install torch==1.13.0a0+git6c9b55e torchvision==0.14.1a0  intel_extension_for_pytorch==1.13.120+xpu -f https://developer.intel.com/ipex-whl-stable-xpu
pip install wandb plotext opencv-python albumentations matplotlib  tabulate shapely
python -m ipykernel install --user --name PT2 --display-name "PT2"


