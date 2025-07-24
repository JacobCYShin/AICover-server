# AICoverGen의 UVR(보컬 분리) 시스템 통합 안내

## 개요

AICoverGen은 python-audio-separator의 핵심 기술을 완전히 통합하여, 외부 의존성 없이 고품질의 보컬/반주 분리(Separation)를 지원합니다. 이 문서는 통합 UVR 시스템의 구조, 사용법, 개발 가이드를 설명합니다.

---

## 아키텍처 구조

```
AICoverGen/
├── src/
│   ├── uvr_integrated/           # 통합 UVR 시스템
│   │   ├── __init__.py
│   │   ├── separator.py          # 메인 Separator 클래스
│   │   ├── architectures/        # 분리 아키텍처별 클래스
│   │   │   ├── __init__.py
│   │   │   ├── common_separator.py
│   │   │   ├── mdx_separator.py  # MDX (구현 완료)
│   │   │   ├── vr_separator.py   # VR (플레이스홀더)
│   │   │   ├── demucs_separator.py # Demucs (플레이스홀더)
│   │   │   └── mdxc_separator.py # MDXC (플레이스홀더)
│   │   └── uvr_lib_v5/           # 오디오 처리 유틸리티
│   │       ├── __init__.py
│   │       ├── spec_utils.py     # 스펙트로그램/정규화 등
│   │       └── stft.py           # STFT 처리
│   ├── uvr_separator.py          # UVR 파이프라인 래퍼
│   ├── main.py                   # 메인 파이프라인
│   └── webui.py                  # WebUI
```

---

## 지원 기능

- **MDX 아키텍처**: ONNX 모델 기반 보컬/반주 분리 (구현 완료)
- **다단계 분리**: 보컬/반주 → 리드/백업 보컬 → 디리버브 → 디노이즈
- **오디오 유틸리티**: STFT, 정규화, 스펙트로그램 변환 등
- **자동 디바이스 감지**: CUDA/MPS/CPU 자동 지원
- **AICoverGen 파이프라인과 완전 통합**
- **향후 확장**: VR, Demucs, MDXC 등 추가 예정

---

## 사용법

### 1. CLI에서 사용

```bash
# UVR 분리 사용 (권장)
python src/main.py -i "노래파일.mp3" -dir "모델폴더명" -p 0 -uvr

# MDX 분리만 사용 (기본)
python src/main.py -i "노래파일.mp3" -dir "모델폴더명" -p 0
```

### 2. WebUI에서 사용

- `python src/webui.py` 실행
- 옵션에서 "UVR 분리 사용" 체크
- 나머지 옵션 설정 후 "Generate" 클릭

### 3. 코드에서 직접 사용

```python
from src.uvr_integrated.separator import Separator
from src.uvr_integrated.architectures.mdx_separator import MDXSeparator

separator = Separator(output_dir="./output")
mdx_separator = MDXSeparator(
    model_path="모델.onnx",
    device="cuda",
    output_dir="./output"
)
output_files = mdx_separator.separate("input.wav")
```

---

## 기술 상세

- **MDX 아키텍처**: ONNX 모델, STFT 기반, 빠른 처리 속도
- **오디오 처리**: 자동 샘플레이트 변환, 정규화, STFT/ISTFT, 후처리
- **디바이스 지원**: CUDA(엔비디아 GPU), MPS(애플 실리콘), CPU
- **모델 종류**: Kim_Vocal_1, UVR_MDXNET_KARA, UVR-De-Echo-Aggressive, UVR-DeNoise 등

---

## 개발/테스트

- `test_integrated_uvr.py`: 클래스/유틸리티 단위 테스트
- `test_uvr_integration.py`: 실제 오디오/모델로 전체 파이프라인 테스트

### 새 아키텍처 추가 방법
1. `architectures/`에 새 분리기 클래스 생성 (예: `demucs_separator.py`)
2. `CommonSeparator` 상속, `separate()` 메서드 구현
3. `separator.py`에 등록

---

## 문제 해결

- **모델 파일 오류**: ONNX 모델 경로/이름 확인
- **메모리 부족**: 배치 크기/오디오 길이 조절, CPU 사용
- **품질 문제**: 입력 오디오 포맷/샘플레이트 확인
- **디버그 모드**: `logging.basicConfig(level=logging.DEBUG)`로 상세 로그 활성화

---

## 향후 로드맵

- VR, Demucs, MDXC 아키텍처 구현
- 모델 자동 다운로드/캐싱
- 실시간 처리/고급 오디오 효과

---

## 라이선스

AICoverGen의 라이선스를 따릅니다. 자세한 내용은 LICENSE 파일 참고. 