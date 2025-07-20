# AICoverGen 배포 가이드

이 문서는 AICoverGen을 Docker와 RunPod 서버리스로 배포하는 방법을 설명합니다.

## 🐳 Docker 로컬 배포

### 1. 사전 요구사항

- Docker 및 Docker Compose 설치
- NVIDIA GPU (선택사항, CPU만으로도 동작)
- NVIDIA Container Toolkit (GPU 사용시)

### 2. 로컬 실행

```bash
# 저장소 클론
git clone <repository-url>
cd AICoverGen

# Docker 이미지 빌드 및 실행
docker-compose up --build

# 백그라운드 실행
docker-compose up -d --build
```

### 3. API 테스트

```bash
# 헬스 체크
curl http://localhost:8000/health

# 사용 가능한 모델 확인
curl http://localhost:8000/models

# API 문서 확인
# 브라우저에서 http://localhost:8000/docs 접속
```

### 4. 클라이언트 예제 실행

```bash
# 클라이언트 예제 실행
python client_example.py
```

## ☁️ RunPod 서버리스 배포

### 1. 사전 준비

1. **RunPod 계정 생성**: https://runpod.io
2. **Docker Hub 계정**: 이미지 푸시용
3. **모델 준비**: RVC 모델들을 준비

### 2. Docker 이미지 빌드 및 푸시

```bash
# Docker Hub에 로그인
docker login

# 이미지 빌드
docker build -t your-username/aicovergen:latest .

# 이미지 푸시
docker push your-username/aicovergen:latest
```

### 3. RunPod 서버리스 템플릿 생성

1. RunPod 대시보드에서 "Serverless" 메뉴로 이동
2. "New Template" 클릭
3. `runpod-template.json` 파일의 내용을 복사하여 설정
4. Docker 이미지 이름을 `your-username/aicovergen:latest`로 변경
5. 템플릿 저장

### 4. 모델 업로드

RunPod 볼륨에 모델들을 업로드:

```bash
# RunPod CLI 설치 (선택사항)
pip install runpod

# 볼륨 생성 및 모델 업로드
# RunPod 대시보드에서 수동으로 진행하는 것을 권장
```

### 5. API 엔드포인트 생성

1. 생성한 템플릿을 선택
2. "Deploy" 클릭
3. 엔드포인트 URL 확인

## 📡 API 사용법

### JSON 요청 형식

```json
{
  "input": {
    "audio_data": "base64_encoded_audio_data",
    "voice_model": "model_name",
    "pitch_change": 0,
    "use_uvr": true,
    "output_format": "mp3"
  }
}
```

### cURL 예제

```bash
# 파일 업로드 방식
curl -X POST "http://your-endpoint-url/generate" \
  -F "audio_file=@song.mp3" \
  -F "voice_model=your_model" \
  -F "pitch_change=0" \
  -F "use_uvr=true" \
  -F "output_format=mp3"

# Base64 방식
curl -X POST "http://your-endpoint-url/generate_base64" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_data": "base64_encoded_audio",
    "voice_model": "your_model",
    "pitch_change": 0,
    "use_uvr": true,
    "output_format": "mp3"
  }'
```

### Python 클라이언트 예제

```python
import requests
import base64

# 오디오 파일을 base64로 인코딩
with open("song.mp3", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode('utf-8')

# API 요청
response = requests.post(
    "http://your-endpoint-url/generate_base64",
    json={
        "audio_data": audio_base64,
        "voice_model": "your_model",
        "pitch_change": 0,
        "use_uvr": True,
        "output_format": "mp3"
    }
)

# 결과 처리
if response.json().get("success"):
    cover_audio = base64.b64decode(response.json()["cover_audio"])
    with open("generated_cover.mp3", "wb") as f:
        f.write(cover_audio)
    print("✅ Cover generated successfully!")
else:
    print(f"❌ Error: {response.json().get('error')}")
```

## 🔧 설정 옵션

### 환경 변수

- `PYTHONPATH`: Python 경로 설정
- `CUDA_VISIBLE_DEVICES`: GPU 설정
- `LOG_LEVEL`: 로그 레벨 설정

### 볼륨 마운트

- `/app/rvc_models`: RVC 음성 모델
- `/app/uvr_models`: UVR 분리 모델
- `/app/song_output`: 출력 파일
- `/app/temp`: 임시 파일

## 🚀 성능 최적화

### GPU 최적화

- CUDA 11.8 사용
- Mixed precision inference
- Batch processing 지원

### 메모리 최적화

- 모델 캐싱
- 임시 파일 자동 정리
- 메모리 효율적인 오디오 처리

## 🔍 모니터링

### 헬스 체크

```bash
curl http://your-endpoint-url/health
```

응답 예시:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_count": 1,
  "available_models": 5
}
```

### 로그 확인

```bash
# Docker 로그
docker-compose logs aicovergen

# RunPod 로그
# RunPod 대시보드에서 확인
```

## 🛠️ 문제 해결

### 일반적인 문제

1. **GPU 메모리 부족**
   - 더 큰 GPU 선택
   - 배치 크기 줄이기

2. **모델 로딩 실패**
   - 모델 파일 경로 확인
   - 볼륨 마운트 확인

3. **오디오 형식 오류**
   - 지원되는 형식: mp3, wav, flac, m4a
   - 파일 크기 제한 확인

### 디버깅

```bash
# 상세 로그 활성화
export LOG_LEVEL=DEBUG

# 컨테이너 내부 접속
docker exec -it aicovergen-api bash
```

## 📊 비용 최적화

### RunPod 비용 절약 팁

1. **적절한 GPU 선택**: 작업에 맞는 GPU 선택
2. **자동 스케일링**: 트래픽에 따른 자동 스케일링
3. **예약 인스턴스**: 장기 사용시 예약 인스턴스 활용

## 🔐 보안 고려사항

1. **API 키 인증**: 프로덕션 환경에서 API 키 사용
2. **Rate Limiting**: 요청 제한 설정
3. **입력 검증**: 모든 입력 데이터 검증
4. **HTTPS**: 프로덕션에서 HTTPS 사용

## 📞 지원

문제가 발생하면 다음을 확인하세요:

1. 로그 파일 확인
2. 시스템 요구사항 충족 여부
3. 모델 파일 정상 여부
4. 네트워크 연결 상태

추가 지원이 필요하면 이슈를 생성해주세요. 