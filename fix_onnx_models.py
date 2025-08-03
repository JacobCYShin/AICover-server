#!/usr/bin/env python3
"""
손상된 ONNX 모델 파일 재다운로드 스크립트
"""

import os
import requests
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# UVR 모델 다운로드 URL들 (예시)
UVR_MODELS = {
    "Kim_Vocal_1.onnx": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Vocal_1.onnx",
    "UVR_MDXNET_KARA.onnx": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR_MDXNET_KARA.onnx",
    # 추가 모델들...
}

def download_model(model_name, url, output_dir):
    """모델 파일 다운로드"""
    output_path = os.path.join(output_dir, model_name)
    
    logger.info(f"다운로드 중: {model_name}")
    logger.info(f"URL: {url}")
    logger.info(f"저장 경로: {output_path}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # 파일 크기 확인
        file_size = os.path.getsize(output_path)
        logger.info(f"✅ 다운로드 완료: {model_name} ({file_size} bytes)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 다운로드 실패: {model_name} - {e}")
        return False

def main():
    """메인 함수"""
    # UVR 모델 디렉토리
    uvr_models_dir = "/runpod-volume/uvr_models"
    
    logger.info("🎵 UVR 모델 파일 복구 시작")
    logger.info("=" * 50)
    
    # 디렉토리 생성
    os.makedirs(uvr_models_dir, exist_ok=True)
    
    # 현재 파일들 확인
    if os.path.exists(uvr_models_dir):
        existing_files = os.listdir(uvr_models_dir)
        logger.info(f"기존 파일들: {existing_files}")
    else:
        logger.info("UVR 모델 디렉토리가 없습니다. 생성합니다.")
    
    # 각 모델 다운로드
    success_count = 0
    for model_name, url in UVR_MODELS.items():
        if download_model(model_name, url, uvr_models_dir):
            success_count += 1
    
    logger.info(f"✅ 완료: {success_count}/{len(UVR_MODELS)} 모델 다운로드 성공")
    
    # 최종 파일 목록 확인
    if os.path.exists(uvr_models_dir):
        final_files = os.listdir(uvr_models_dir)
        logger.info(f"최종 파일들: {final_files}")

if __name__ == "__main__":
    main() 