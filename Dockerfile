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

# 필요한 requirements 설치 - 캐시 무시하고 강제 재설치
# (onnxruntime-gpu는 이미 CUDA 12 호환 버전으로 설치됨)
RUN pip install --upgrade "pip<24.0" && \
    pip install --no-cache-dir --force-reinstall -r /app/AICover-server/builder/requirements.txt

# cuDNN 라이브러리 심볼릭 링크 (onnxruntime이 찾을 수 있도록) - simplified version from Dockerfile.ubuntu
RUN ln -s /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/libcudnn.so.9 /usr/lib/libcudnn.so.9 && \
    ln -s /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/libcudnn.so.9 /usr/lib/x86_64-linux-gnu/libcudnn.so.9 && \
    ln -s /usr/local/cuda/lib64/libcublas.so.12 /usr/lib/x86_64-linux-gnu/libcublas.so.12 && \
    ldconfig

# 작업 디렉토리를 AICover-server로 변경 (handler.py가 run.py를 찾을 수 있도록)
WORKDIR /app/AICover-server

# 불필요한 파일들 정리 (.ipynb_checkpoints, .DS_Store 등)
RUN find /app/AICover-server -name ".*" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /app/AICover-server -name ".DS_Store" -type f -delete 2>/dev/null || true

# RunPod 네트워크 볼륨 마운트 포인트 준비
# rvc_models와 uvr_models 모두 /runpod-volume/에서 마운트됨
RUN rm -rf /app/AICover-server/rvc_models && \
    rm -rf /app/AICover-server/uvr_models && \
    mkdir -p /runpod-volume/rvc_models && \
    mkdir -p /runpod-volume/uvr_models && \
    ln -s /runpod-volume/rvc_models /app/rvc_models && \
    ln -s /runpod-volume/uvr_models /app/AICover-server/uvr_models

# UVR 모델 디렉토리 환경변수 설정 (심볼릭 링크된 경로 사용)
ENV AUDIO_SEPARATOR_MODEL_DIR=/app/AICover-server/uvr_models

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