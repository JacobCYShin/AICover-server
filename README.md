# 🎤 AICoverGen: AI 커버 생성 파이프라인

AICoverGen은 RVC v2 기반 AI 보이스 모델과 통합 UVR(보컬 분리) 시스템을 활용해, 유튜브/로컬 오디오에서 원하는 가수 목소리로 커버곡을 자동 생성하는 오픈소스 파이프라인입니다.

---

## 주요 특징

- **RVC v2 기반 AI 보이스 변환**: 다양한 캐릭터/가수 목소리 지원
- **UVR(보컬 분리) 완전 통합**: MDX, VR, Demucs, MDXC 등 다단계 분리
- **Docker & RunPod 서버리스 지원**: 원클릭 배포, 클라우드 API 제공
- **WebUI & REST API**: 웹에서 손쉽게 사용, JSON 기반 API 제공
- **고품질 오디오 출력**: mp3/wav 지원, 피치/볼륨/리버브 등 세부 옵션
- **확장성**: 새로운 분리 모델 및 보이스 모델 손쉽게 추가 가능

---

## 설치 및 실행

### 1. Docker로 로컬 실행

```bash
git clone <레포주소>
cd AICoverGen
docker-compose up --build
```
- API: [http://localhost:8000](http://localhost:8000)
- WebUI: [http://localhost:7860](http://localhost:7860)

### 2. RunPod 서버리스 배포

1. Docker 이미지 빌드 및 푸시
    ```bash
    docker build -t <your-dockerhub>/aicovergen:latest .
    docker push <your-dockerhub>/aicovergen:latest
    ```
2. RunPod 대시보드에서 템플릿 생성 (`runpod-template.json` 참고)
3. 모델 볼륨 업로드 및 서버리스 엔드포인트 배포

자세한 내용은 [DEPLOYMENT.md](DEPLOYMENT.md) 참고

---

## 사용법

### 1. WebUI

```bash
python src/webui.py
```
- 모델 다운로드/업로드, 옵션 설정, 커버 생성 모두 웹에서 가능

### 2. CLI

```bash
python src/main.py -i "노래파일.mp3" -dir "모델폴더명" -p 0 -uvr
```
- `-uvr` 옵션으로 고품질 UVR 분리 사용

### 3. REST API

#### 커버 생성 (Base64 방식)
```bash
curl -X POST "http://localhost:8000/generate_base64" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_data": "<base64오디오>",
    "voice_model": "모델명",
    "pitch_change": 0,
    "use_uvr": true,
    "output_format": "mp3"
  }'
```

#### 커버 생성 (파일 업로드)
```bash
curl -X POST "http://localhost:8000/generate" \
  -F "audio_file=@노래.mp3" \
  -F "voice_model=모델명" \
  -F "pitch_change=0" \
  -F "use_uvr=true" \
  -F "output_format=mp3"
```

#### 사용 가능한 모델 조회
```bash
curl http://localhost:8000/models
```

---

## 폴더 구조

```
AICoverGen/
  builder/requirements.txt      # Python 의존성
  src/
    handler.py                 # RunPod 서버리스 엔트리포인트
    api_server.py              # FastAPI 서버 (로컬/테스트용)
    main.py                    # 커버 생성 파이프라인
    webui.py                   # WebUI
    uvr_integrated/            # 통합 UVR 분리 시스템
  rvc_models/                  # 보이스 모델 폴더
  uvr_models/                  # 분리 모델 폴더
  song_output/                 # 결과 오디오 저장
  Dockerfile
  docker-compose.yml
  runpod-template.json
  DEPLOYMENT.md
  README.md
```

---

## 테스트

- `test_uvr_integration.py`: 실제 오디오/모델로 전체 파이프라인 통합 테스트
- `test_integrated_uvr.py`: UVR 내부 클래스/유틸리티 단위 테스트

---

## 자주 묻는 질문

- **Q. GPU 없이도 동작하나요?**  
  A. CPU로도 동작하지만, GPU에서 훨씬 빠릅니다.

- **Q. 지원 오디오 포맷은?**  
  A. mp3, wav, flac 등 대부분 지원

- **Q. 모델은 어디서 구하나요?**  
  A. [AI Hub Discord](https://discord.gg/aihub) 등에서 다운로드, WebUI로 업로드 가능

---

## 라이선스 및 주의사항

- 비상업적, 연구/개인용으로만 사용하세요.
- 타인의 목소리/저작권 침해에 주의하세요.
- 자세한 이용 제한은 원본 README의 Terms of Use 참고

---

## 문의/기여

- 이슈/PR 환영합니다!
- 추가 문의는 GitHub Issue 또는 이메일로 연락주세요.
