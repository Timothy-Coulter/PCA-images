# Use an NVIDIA CUDA base image with Ubuntu 22.04 and CUDA 12.1
# Check NVIDIA's NGC catalog or Docker Hub for the latest appropriate tags
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH
ENV NB_USER vscode
ENV NB_UID 1000
ENV HOME /home/$NB_USER
ENV NB_GID 100

# Create a non-root user 'vscode' and grant necessary permissions
RUN groupadd --gid $NB_GID $NB_USER && \
    useradd --uid $NB_UID --gid $NB_GID --shell /bin/bash --create-home $NB_USER && \
    mkdir -p $CONDA_DIR && \
    chown -R $NB_UID:$NB_GID $CONDA_DIR && \
    mkdir -p /sudoers.d && \
    echo $NB_USER ALL=\(root\) NOPASSWD:ALL > /sudoers.d/$NB_USER && \
    chmod 0440 /sudoers.d/$NB_USER

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
    curl \
    unzip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Switch to the non-root user
USER $NB_UID

# Install Anaconda
WORKDIR /tmp
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash ./miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    $CONDA_DIR/bin/conda init bash && \
    # Temporarily switch back to root to fix permissions potentially changed by conda init
    sudo chown -R $NB_UID:$NB_GID $HOME

# Activate conda environment and install packages
# Using pip within the conda environment for TF and PyTorch GPU often works well
# Ensure compatibility between CUDA version (12.1), cuDNN (8), TF, and PyTorch
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> $HOME/.bashrc && \
    echo "conda activate base" >> $HOME/.bashrc && \
    conda install -y python=3.10 jupyterlab numpy pandas scikit-learn matplotlib seaborn opencv pillow && \
    # Install PyTorch with CUDA 12.1 support (check official PyTorch website for the correct command for your CUDA version)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    # Install TensorFlow with GPU support (requires CUDA and cuDNN compatible versions)
    # The '[and-cuda]' variant often helps pull necessary components.
    pip install tensorflow[and-cuda] && \
    # Install other useful libraries
    pip install requests tqdm kaggle fiftyone && \
    # Clean up conda cache
    conda clean -afy

# Set back to the default working directory
WORKDIR $HOME

# Expose Jupyter port if needed (optional)
# EXPOSE 8888

# Default command (optional, can be overridden in devcontainer.json)
# CMD ["/bin/bash"]
