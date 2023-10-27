# Forest Fire Prediction

### Learning Objectives:

- Apply Intel Extension for PyTorch to convert model and data to xpu 
- Apply fine tuning to satellite images to predict fire fires 2 years in advance
- Generate synthetic satellite images using Stable Diffusion
- Optimize a four-stage Stable Diffusion pipeline using Intel(r) Extensions for PyTorch*

# First - a No Code approach: Stable Diffusion

- ![Stable Diffusion Exmaple](assets/StableDiffExample.png)
- Run all cells
- When for an image url try: https://github.com/intelsoftware/ForestFirePrediction/blob/main/data/real_USGS_NAIP/train/Fire/m_3912105_sw_10_h_20160713.png?raw=true
- ![ImageURL](assets/ImageURL.png)
- For text prompt, try: similar aerial photo just translate location

## To Access via Intel(R) Developer Cloud:

1. Go to the Intel Developer Cloud by scanning the QR code or by visiting
https://cloud.intel.com/
- ![Intel Developer Cloud Jupyter Notebook](assets/qrcode_console.idcservice.net.png)

2. Click “Get Started”
3. Subscribe to the “Standard – Free” service tier and complete your cloud
registration.
- ![Standard Free](assets/StandardFree.png)
4. To start up a free and quick Jupyter notebook session with the latest **4th Gen
Intel Xeon CPU** and **Intel Data Center GPU Max 1100**, click the “Training and
Workshops” icon and then “Launch JupyterLab”, or one of the specific training
materials launches.
- ![Training Jupyter Launch](assets/TrainingJupyter.png)
5. Navigate using the Jupyter Hub folder in the left pane and Launch a terminal
  - ![Terminal](assets/Launcher.png)
6. Git Clone the repo
- git clone https://github.com/IntelSoftware/ForestFirePrediction.git
7. cd to ForestFirePrediction
8. conda activate pytorch-gpu
9. pip install -r requirements.txt
10. Open and run each Jupyter Notebook in sequence
  
## Notices and Disclaimers

Intel technologies may require enabled hardware, software or service activation.

No product or component can be absolutely secure. 

Your costs and results may vary. 

© Intel Corporation. Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries. Other names and brands may be claimed as the property of others. 
