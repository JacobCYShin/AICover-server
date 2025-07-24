# Dockerfile

FROM audio-separator-gpu:latest 

WORKDIR /app

# AICover-server 복사
COPY . /app/AICover-server

RUN apt-get update && apt-get install -y sox

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# 필요한 requirements 설치
RUN pip install --upgrade "pip<24.0" && \
    pip install --no-cache-dir -r /app/AICover-server/builder/requirements.txt && \
    pip uninstall -y onnxruntime-gpu && \
    pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# cuDNN 라이브러리 심볼릭 링크 (onnxruntime이 찾을 수 있도록)
RUN ln -s /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/libcudnn.so.9 /usr/lib/libcudnn.so.9 && \
    ln -s /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/libcudnn.so.9 /usr/lib/x86_64-linux-gnu/libcudnn.so.9 && \
    ln -s /usr/local/cuda/lib64/libcublas.so.12 /usr/lib/x86_64-linux-gnu/libcublas.so.12 && \
    ldconfig

# entrypoint.sh 복사 및 실행 권한 부여
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 기본 실행 명령: entrypoint.sh → handler.py
ENTRYPOINT ["/entrypoint.sh"]
# RunPod serverless 기본 진입점
CMD ["python3", "-u", "handler.py"]