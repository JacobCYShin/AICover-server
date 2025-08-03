# Dockerfile

# Use the latest CUDA 12 runtime as base image
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /workdir

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    python3 \
    python3-pip \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install core dependencies
RUN python3 -m pip install --upgrade pip

# Install audio-separator with caching
RUN --mount=type=cache,target=/root/.cache \
    pip3 install "audio-separator"

# Install RunPod dependency
RUN pip install runpod

# Set up application directory
WORKDIR /app

# Copy AICover-server application
COPY . /app/AICover-server

# Set environment variables
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
ENV AUDIO_SEPARATOR_MODEL_DIR=/app/AICover-server/uvr_models

# Install Python dependencies from requirements.txt
RUN pip install --upgrade "pip<24.0" && \
    pip install --no-cache-dir --force-reinstall -r /app/AICover-server/builder/requirements.txt

# Set up cuDNN library symlinks for ONNX Runtime
RUN ln -s /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/libcudnn.so.9 /usr/lib/libcudnn.so.9 && \
    ln -s /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/libcudnn.so.9 /usr/lib/x86_64-linux-gnu/libcudnn.so.9 && \
    ln -s /usr/local/cuda/lib64/libcublas.so.12 /usr/lib/x86_64-linux-gnu/libcublas.so.12 && \
    ldconfig

# Clean up unnecessary files (.ipynb_checkpoints, .DS_Store, etc.)
RUN find /app/AICover-server -name ".*" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /app/AICover-server -name ".DS_Store" -type f -delete 2>/dev/null || true

# Set up RunPod volume mount points
# rvc_models and uvr_models are mounted from /runpod-volume/
RUN rm -rf /app/AICover-server/rvc_models && \
    rm -rf /app/AICover-server/uvr_models && \
    mkdir -p /runpod-volume/rvc_models && \
    mkdir -p /runpod-volume/uvr_models && \
    ln -s /runpod-volume/rvc_models /app/rvc_models && \
    ln -s /runpod-volume/uvr_models /app/AICover-server/uvr_models

# Create output directories
RUN mkdir -p /app/AICover-server/song_output && \
    mkdir -p /app/AICover-server/temp_uploads && \
    mkdir -p /app/temp

# Set permissions
RUN chmod -R 755 /app/AICover-server

# Change to AICover-server directory
WORKDIR /app/AICover-server

# RunPod serverless entry point
CMD ["python3", "-u", "src/handler.py"]