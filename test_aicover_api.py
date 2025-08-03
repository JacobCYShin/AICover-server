## AICover-server RunPod API 테스트 코드
# Colab에서 사용 가능한 테스트 스니펫

import requests
import base64
import json
import time
from IPython.display import Audio, display
import io

# ====== 사용자 설정 ======
API_KEY = "your_runpod_api_key_here"  # RunPod API 키 입력
ENDPOINT_ID = "your_endpoint_id_here"  # RunPod Endpoint ID 입력

# ====== 테스트 설정 ======
AUDIO_FILE_PATH = "input_song.mp3"  # 입력 오디오 파일 경로
VOICE_MODEL = "Jungkook"  # 사용할 보이스 모델명
PITCH_CHANGE = 0  # 피치 변경 (-12 ~ +12)
USE_UVR = True  # UVR 분리 사용 여부
OUTPUT_FORMAT = "mp3"  # 출력 포맷 (mp3, wav)

# ====== 함수 정의 ======
def encode_audio_to_base64(file_path):
    """오디오 파일을 base64로 인코딩"""
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")

def decode_base64_to_audio(base64_string, output_path):
    """base64를 오디오 파일로 디코딩"""
    audio_data = base64.b64decode(base64_string)
    with open(output_path, "wb") as f:
        f.write(audio_data)
    return output_path

def test_aicover_api():
    """AICover API 테스트"""
    
    # 1. 입력 오디오 → base64 인코딩
    print("🎵 오디오 파일 인코딩 중...")
    try:
        encoded_audio = encode_audio_to_base64(AUDIO_FILE_PATH)
        print(f"✅ 인코딩 완료 (크기: {len(encoded_audio)} chars)")
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {AUDIO_FILE_PATH}")
        return
    
    # 2. API 요청 구성
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "audio_data": encoded_audio,
            "voice_model": VOICE_MODEL,
            "pitch_change": PITCH_CHANGE,
            "use_uvr": USE_UVR,
            "output_format": OUTPUT_FORMAT
        }
    }
    
    # 3. 요청 전송
    print("🚀 API 요청 전송 중...")
    print(f"📊 요청 정보:")
    print(f"   - 보이스 모델: {VOICE_MODEL}")
    print(f"   - 피치 변경: {PITCH_CHANGE}")
    print(f"   - UVR 사용: {USE_UVR}")
    print(f"   - 출력 포맷: {OUTPUT_FORMAT}")
    
    start_time = time.time()
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=300)  # 5분 타임아웃
        
        if response.status_code != 200:
            print(f"❌ 요청 실패: {response.status_code}")
            print(f"응답: {response.text}")
            return
            
    except requests.exceptions.Timeout:
        print("❌ 요청 타임아웃 (5분 초과)")
        return
    except Exception as e:
        print(f"❌ 요청 중 오류: {str(e)}")
        return
    
    processing_time = time.time() - start_time
    print(f"⏱️ 처리 시간: {processing_time:.2f}초")
    
    # 4. 응답 결과 처리
    try:
        result = response.json()
        print("✅ 응답 수신 완료")
        
        if "error" in result:
            print(f"❌ 서버 에러: {result['error']}")
            return
            
        # 출력 오디오 추출
        if "output" in result and "output_audio" in result["output"]:
            output_base64 = result["output"]["output_audio"]
            
            # 5. 디코딩 및 저장
            output_filename = f"ai_cover_output.{OUTPUT_FORMAT}"
            decode_base64_to_audio(output_base64, output_filename)
            
            print(f"🎉 결과 저장 완료 → {output_filename}")
            
            # Colab에서 오디오 재생
            display(Audio(output_filename))
            
            # 추가 정보 출력
            if "metadata" in result["output"]:
                metadata = result["output"]["metadata"]
                print(f"\n📋 처리 정보:")
                for key, value in metadata.items():
                    print(f"   - {key}: {value}")
                    
        else:
            print("❌ 예상된 출력 형식이 아닙니다.")
            print(f"응답 구조: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
    except json.JSONDecodeError:
        print("❌ JSON 디코딩 실패")
        print(f"응답 텍스트: {response.text}")

# ====== 고급 테스트 함수 ======
def test_multiple_models():
    """여러 모델로 테스트"""
    models = ["Jungkook", "IU", "NewJeans"]  # 사용 가능한 모델들
    
    for model in models:
        print(f"\n🎤 {model} 모델 테스트 중...")
        global VOICE_MODEL
        VOICE_MODEL = model
        test_aicover_api()
        time.sleep(2)  # 요청 간 간격

def test_pitch_variations():
    """다양한 피치로 테스트"""
    pitches = [-2, 0, 2]  # 낮음, 원본, 높음
    
    for pitch in pitches:
        print(f"\n🎵 피치 {pitch:+d} 테스트 중...")
        global PITCH_CHANGE
        PITCH_CHANGE = pitch
        test_aicover_api()
        time.sleep(2)

# ====== 실행 ======
if __name__ == "__main__":
    print("🎵 AICover-server API 테스트 시작")
    print("="*50)
    
    # 기본 테스트
    test_aicover_api()
    
    # 추가 테스트 (주석 해제하여 사용)
    # test_multiple_models()
    # test_pitch_variations() 