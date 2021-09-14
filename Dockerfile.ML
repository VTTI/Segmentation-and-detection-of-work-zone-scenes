# NVIDIA CUDA image as a base
# We also mark this image as "ml-base" so we could use it by name
FROM nvidia/cuda:10.2-runtime AS ml-base

# Install Python and its tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools
RUN pip3 -q install pip --upgrade
# Install all basic packages
RUN pip3 install \
    # Jupyter itself
    jupyter \
    # Numpy and Pandas are required a-priori
    numpy pandas \
    # PyTorch with CUDA 10.2 support and Torchvision
    torch torchvision \
    # Upgraded version of Tensorboard with more features
    tensorboardX

ENV CUDA_VISIBLE_DEVICES 0,1,2,3
RUN export CUDA_VISIBLE_DEVICES=0,1,2,3
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get install vim -y
# Install additional packages
RUN pip3 install opencv-python Shapely matplotlib scipy tqdm scikit_learn tensorboard pillow jupyterlab PyYAML segmentation-models-pytorch
WORKDIR /opt/app


## -------------
## Add alias to ensure calling python calls python3
## -------------
RUN echo 'alias python="python3"' >> ~/.bashrc
RUN echo 'alias jupyter="jupyter lab --port=8888 --no-browser --ip=0.0.0.0 --allow-root"' >> ~/.bashrc
ENTRYPOINT /bin/bash  && cd /opt/app
