ARG BASE_TORCH_VERSION=2.2.0
ARG BASE_CUDA_VERSION=11.8
ARG BASE_CUDNN_VERSION=8

FROM docker.io/pytorch/pytorch:${BASE_TORCH_VERSION}-cuda${BASE_CUDA_VERSION}-cudnn${BASE_CUDNN_VERSION}-devel

RUN apt update -y && \
    DEBIAN_FRONTEND=noniteractive apt install -y \
    libopencv-dev \
    git

ARG UID=1000
ARG USERNAME=torchuser
RUN useradd -l -m -u ${UID} ${USERNAME}
USER ${UID}
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

# install pip modules
COPY ./docker/torch/requirements.txt requirements.txt
RUN pip --default-timeout=100 install --user -r requirements.txt
