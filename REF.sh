

# 빌드 방법
docker build -t aicover-server-with-separator -f Dockerfile.aicover .

# 컨테이너 bash 실행 방법 (바로 삭제하고 싶으면 --rm을 붙여야 함)
docker run --rm -it --gpus all   -v $(pwd):/workspace   -w /app/AICover-server   --entrypoint /bin/bash   aicover-server-with-separator

# 컨테이너 내부에서 실행 방법
python3 run.py --rvc_dirname Jimin --song_input input.wav --pitch_change_all 0

# audio separator 실행 방법
audio-separator /app/AICover-server/input.wav --output_dir /app/AICover-server/output_sep

import onnxruntime as ort
print('Providers:', ort.get_available_providers())
sess = ort.InferenceSession('/workspace/uvr_models/Kim_Vocal_1.onnx', providers=['CUDAExecutionProvider'])
print('Using:', sess.get_providers())


ln -s /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/libcudnn_graph.so.9 /usr/lib/libcudnn_graph.so.9
ln -s /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/libcudnn_graph.so.9 /usr/lib/x86_64-linux-gnu/libcudnn_graph.so.9
ldconfig

