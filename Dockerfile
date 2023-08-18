FROM nvidia/cuda:11.3.0-runtime-ubuntu18.04

RUN apt-key del 7fa2af80
ADD https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb .
RUN dpkg -i cuda-keyring_1.0-1_all.deb 
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt update

RUN apt-get update && apt-get install -y build-essential curl git wget gosu sudo
RUN apt-get install -y build-essential libpq-dev libssl-dev openssl libffi-dev zlib1g-dev
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y python3.8
RUN apt-get install -y python3.8-dev
RUN apt-get install -y libjpeg-dev
RUN python3.8 -m pip install -U pip
RUN python3.8 -m pip install cython
RUN python3.8 -m pip install numpy
RUN python3.8 -m pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN python3.8 -m pip install torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN python3.8 -m pip install torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN python3.8 -m pip install pyyaml
RUN python3.8 -m pip install pandas
RUN python3.8 -m pip install IPython
RUN python3.8 -m pip install matplotlib
RUN python3.8 -m pip install scipy

# ==============

ARG USERNAME=docker
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

