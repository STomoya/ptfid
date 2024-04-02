ARG BASE_TORCH_VERSION
ARG BASE_CUDA_VERSION
ARG BASE_CUDNN_VERSION

FROM pytorch/pytorch:${BASE_TORCH_VERSION}-cuda${BASE_CUDA_VERSION}-cudnn${BASE_CUDNN_VERSION}-devel

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
