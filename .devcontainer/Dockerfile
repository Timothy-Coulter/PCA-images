FROM ubuntu:24.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install essential packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    vim \
    software-properties-common \
    ca-certificates \
    gnupg \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Setup for NVIDIA CUDA toolkit - temporarily disabled for testing
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
#     dpkg -i cuda-keyring_1.1-1_all.deb && \
#     rm cuda-keyring_1.1-1_all.deb && \
#     apt-get update && apt-get install -y \
#     cuda-toolkit-11-8 \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables - temporarily disabled for testing
# ENV CUDA_HOME=/usr/local/cuda
# ENV PATH=${CUDA_HOME}/bin:${PATH}
# ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Use system Python instead of Conda
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Ensure pip is up to date
RUN python3 -m pip install --upgrade pip

# Install TensorFlow (CPU version for testing)
RUN pip install tensorflow==2.15.*

# Install PyTorch (CPU version for testing)
RUN pip install torch==2.1.* torchvision==0.16.* torchaudio==2.1.*

# Install Jupyter and essential data science packages
RUN pip install jupyter pandas numpy matplotlib seaborn scikit-learn scipy statsmodels 

# Install visualization libraries
RUN pip install plotly dash bokeh altair

# Install data handling libraries
RUN pip install dask pyarrow h5py joblib pyyaml tqdm 

# Install ML libraries and tools
RUN pip install xgboost lightgbm catboost optuna hyperopt mlflow wandb 

# Install deep learning extensions
RUN pip install transformers datasets huggingface_hub tensorflow-datasets keras-cv keras-nlp

# Install useful development tools
RUN pip install black flake8 pylint pytest ipywidgets jupyterlab

# Create a non-root user matching the remoteUser from devcontainer.json
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set up environment for non-root user
RUN mkdir -p /home/$USERNAME/.cache/pip && \
    chown -R $USERNAME:$USERNAME /home/$USERNAME/.cache/pip

# Setup Python for user
RUN echo "export PATH=\$PATH:\$HOME/.local/bin" >> /home/$USERNAME/.bashrc && \
    chown $USERNAME:$USERNAME /home/$USERNAME/.bashrc

# Clean up unnecessary cache files
RUN pip cache purge

WORKDIR /workspace

# Reset environment variable
ENV DEBIAN_FRONTEND=dialog

# Set default shell to bash
SHELL ["/bin/bash", "-c"]