ARG CUDA_IMAGE_TAG=11.8.0
ARG OS_VERSION=22.04
ARG USER_ID=1000
# Define base image.
FROM nvidia/cuda:${CUDA_IMAGE_TAG}-devel-ubuntu${OS_VERSION}
ARG CUDA_IMAGE_TAG
ARG OS_VERSION
ARG USER_ID

# metainformation
LABEL org.opencontainers.image.version = "0.1.18"
LABEL org.opencontainers.image.source = "https://github.com/nerfstudio-project/nerfstudio"
LABEL org.opencontainers.image.licenses = "Apache License 2.0"
LABEL org.opencontainers.image.base.name="docker.io/library/nvidia/cuda:${CUDA_IMAGE_TAG}-devel-ubuntu${OS_VERSION}"

# Variables used at build time.
## CUDA architectures, required by Colmap and tiny-cuda-nn.
## NOTE: All commonly used GPU architectures are included and supported here. To speedup the image build process remove all architectures but the one of your explicit GPU. Find details here: https://developer.nvidia.com/cuda-gpus (8.6 translates to 86 in the line below) or in the docs.
ARG CUDA_ARCHITECTURES=86;80;75

# Set environment variables.
## Set non-interactive to prevent asking for user inputs blocking image creation.
ENV DEBIAN_FRONTEND=noninteractive
## Set timezone as it is required by some packages.
ENV TZ=Europe/Berlin
## CUDA Home, required to find CUDA in some packages.
ENV CUDA_HOME="/usr/local/cuda"

# Install required apt packages and clear cache afterwards.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    git \
    nano \
    sudo \
    vim-tiny \
    wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /programs

ARG PYTHON_VERSION=3.10
ARG PYTORCH_VERSION=2.0.1
ARG TORCHVISION_VERSION=0.15.2
ARG CUDA=11.8

RUN export CUDA_SHORT=$(echo "$CUDA" | sed "s/[.]//") && \
     curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     bash ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION cython && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda$(echo "$CUDA" | sed "s/[.]//") && \
     /opt/conda/bin/conda clean -ya && \
     rm -rf /root/.cache/
ENV PATH /opt/conda/bin:$PATH

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# now use pip to install to avoid conda install all cuda stuff again, and clean cache afterwards (it's ~3G cache!)
RUN export CUDA_SHORT=$(echo "$CUDA" | sed "s/[.]//") && \
    pip install --no-cache-dir mkl-include==2023.0.0 && \
    pip install --no-cache-dir torch==${PYTORCH_VERSION}+cu$CUDA_SHORT torchvision==${TORCHVISION_VERSION}+cu$CUDA_SHORT --extra-index-url https://download.pytorch.org/whl/cu$CUDA_SHORT && \
    rm -rf /root/.cache/

#nerfw
RUN pip install kornia einops torch_optimizer lightning\<2.1.0

# Create non root user and setup environment.
RUN useradd -m -d /programs -g root -G sudo -u ${USER_ID} user
RUN usermod -aG sudo user
# Set user password
RUN echo "user:user" | chpasswd
# Ensure sudo group users are not asked for a password when using sudo command by ammending sudoers file
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers


# Switch to new uer and workdir.
USER root
RUN chmod -R a+w /programs
RUN chown -R  ${USER_ID} /programs
USER ${USER_ID}
WORKDIR /programs
#RUN chmod -R a+w /programs
# Add local user binary folder to PATH variable.
ENV PATH="${PATH}:/programs/.local/bin"
#SHELL ["/bin/bash", "-c"]

# Change working directory
WORKDIR /workspace
