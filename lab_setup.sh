#!/bin/bash

error_exit() {
    echo "$1" 1>&2
    exit 1
}

setup_oneapi() {
    source /opt/intel/oneapi/setvars.sh --force || error_exit "Failed to set up Intel OneAPI."
}

create_conda_environment() {
    ENV_NAME=${1:-PT}
    conda config --add channels intel || error_exit "Failed to add Intel channel."
    conda create -y -n "$ENV_NAME" intelpython3_full || error_exit "Failed to create Conda environment."
    source $(conda info --base)/etc/profile.d/conda.sh || error_exit "Failed to source Conda profile."
    conda activate "$ENV_NAME" || error_exit "Failed to activate Conda environment."
}

install_packages() {
    conda install -y -c conda-forge jupyterlab ipywidgets || error_exit "Failed to install Jupyter and widgets."
    conda install -y -c anaconda seaborn pandas || error_exit "Failed to install Seaborn and Pandas."
    pip install torch==1.13.0a0+git6c9b55e torchvision==0.14.1a0 intel_extension_for_pytorch==1.13.120+xpu \
        -f https://developer.intel.com/ipex-whl-stable-xpu || error_exit "Failed to install PyTorch and Intel extension."
    pip install protobuf tensorboardX || error_exit "Failed to install protobuf tensorboardX packages."
    pip install plotext opencv-python albumentations tabulate shapely diffusers transformers || error_exit "Failed to install additional packages."
}

prepare_data() {
    python 02_Prepare_Data.py || error_exit "Failed to prepare data."
}

download_models() {
    python download_models.py || error_exit "Failed to download models."
}

main() {
    prepare_env=true
    prepare_data=true
    download_models=true
    for arg in "$@"; do
        case $arg in
            --prepare_env) prepare_env=true ;;
            --prepare_data) prepare_data=true ;;
            --download_models) download_models=true ;;
            *) error_exit "Unknown argument: $arg" ;;
        esac
    done
    if [ "$prepare_env" = true ]; then
        setup_oneapi
        create_conda_environment "PT"
        install_packages
    fi
    if [ "$prepare_data" = true ]; then
        prepare_data
    fi
    if [ "$download_models" = true ]; then
        download_models
    fi
    echo "Script execution successful!"
}

main "$@"
