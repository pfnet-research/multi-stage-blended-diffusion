FROM pytorch/pytorch

RUN apt-get update; apt-get install git -y
RUN pip install albumentations opencv-python pudb imageio pytorch-lightning imageio-ffmpeg omegaconf test-tube streamlit einops torch-fidelity transformers
RUN pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
RUN pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip