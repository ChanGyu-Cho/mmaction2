FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    curl \
    wget \
    ca-certificates \
    python3-pip \
    python3-dev \
    python3-setuptools \
    ninja-build \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu116

RUN pip install -U openmim && \
    mim install mmengine==0.7.4 && \
    mim install mmcv==2.1.0

WORKDIR /workspace
RUN git clone https://github.com/open-mmlab/mmaction2.git
WORKDIR /workspace/mmaction2
RUN pip install -e .

RUN python -c "import torch; import mmcv; import mmaction; print('✅ All dependencies are installed correctly.')"

WORKDIR /workspace/mmaction2
