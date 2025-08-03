# Dockerfile

# Use the latest CUDA 12 runtime as base image (from Dockerfile.audio-separator)
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /workdir

# Install basic dependencies (from Dockerfile.audio-separator)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    python3 \
    python3-pip \
    sox \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# Install the CUDA 12 compatible version of ONNXRuntime (from Dockerfile.audio-separator)
# See https://onnxruntime.ai/docs/install/
RUN pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# Install audio-separator without any specific onnxruntime (from Dockerfile.audio-separator)
RUN --mount=type=cache,target=/root/.cache \
    pip3 install "audio-separator"

# Install RunPod dependency early
RUN pip install runpod

# Set up AICover-server application
WORKDIR /app

# AICover-server 복사
COPY . /app/AICover-server

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# 필요한 requirements 설치
RUN pip install --upgrade "pip<24.0" && \
    pip install --no-cache-dir -r /app/AICover-server/builder/requirements.txt && \
    pip uninstall -y onnxruntime-gpu && \
    pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# cuDNN 라이브러리 심볼릭 링크 (onnxruntime이 찾을 수 있도록) - simplified version from Dockerfile.ubuntu
RUN ln -s /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/libcudnn.so.9 /usr/lib/libcudnn.so.9 && \
    ln -s /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/libcudnn.so.9 /usr/lib/x86_64-linux-gnu/libcudnn.so.9 && \
    ln -s /usr/local/cuda/lib64/libcublas.so.12 /usr/lib/x86_64-linux-gnu/libcublas.so.12 && \
    ldconfig

# 작업 디렉토리를 AICover-server로 변경 (handler.py가 run.py를 찾을 수 있도록)
WORKDIR /app/AICover-server

# UVR 모델 디렉토리 미리 생성 및 환경변수 설정
RUN mkdir -p /app/AICover-server/uvr_models
ENV AUDIO_SEPARATOR_MODEL_DIR=/app/AICover-server/uvr_models

# RunPod 네트워크 볼륨 마운트 포인트 준비 - 경로 수정
# /runpod-volume/rvc_models → /app/rvc_models 연결을 위한 심볼릭 링크 (src/handler.py가 /app/rvc_models를 참조)
RUN rm -rf /app/AICover-server/rvc_models && \
    mkdir -p /runpod-volume/rvc_models && \
    ln -s /runpod-volume/rvc_models /app/rvc_models

# 선택사항: UVR 모델들을 빌드 시 미리 다운로드 (Cold start 최적화)
# 주석을 해제하면 빌드 시 다운로드됩니다
# RUN python3 download_uvr_models.py

# 출력 디렉토리 미리 생성
RUN mkdir -p /app/AICover-server/song_output && \
    mkdir -p /app/AICover-server/temp_uploads && \
    mkdir -p /app/temp

# 권한 설정
RUN chmod -R 755 /app/AICover-server

# RunPod serverless 기본 진입점 - src/handler.py 사용 (RunPod 호환)
CMD ["python3", "-u", "src/handler.py"]