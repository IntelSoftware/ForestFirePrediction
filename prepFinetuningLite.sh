#! /bin/bash

echo "Gathering data"
cp /data/oneapi_workshop/big_datasets/jurassic/*.zip .
unzip -qq data.zip
echo "Gathering Models"
mv models.zip models0.zip
wget https://www.dropbox.com/s/tj8j6infag4sfcf/models.zip
unzip -qq models.zip
echo "Setting conda environment"
source /glob/development-tools/versions/oneapi/2022.2/oneapi/setvars.sh --force
#python -m ipykernel install --user --name base --display-name "base"
conda activate base
pip install -r requirements.txt
# echo "creating new environment"
# conda create --clone pytorch --name openvinopytorch2
# echo "Activating environment"
# conda activate openvinopytorch2
# echo "Instalingl openvino-dev"
# ~/.conda/envs/openvinopytorch/bin/pip install --user openvino-dev[pytorch]
# echo "Registering new kernel in Jupyter"
# python -m ipykernel install --user --name openvinopytorch2
# echo "Installing OpenVINO lan requirements"
# pip install -r requirements_openvino.txt
conda activate base
