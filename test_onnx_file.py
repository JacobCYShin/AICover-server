#!/usr/bin/env python3
"""
ONNX 파일 무결성 테스트 스크립트
"""

import os
import sys
import onnx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_onnx_file(file_path):
    """ONNX 파일 무결성 테스트"""
    try:
        logger.info(f"테스트 중인 파일: {file_path}")
        
        # 파일 존재 확인
        if not os.path.exists(file_path):
            logger.error(f"파일이 존재하지 않습니다: {file_path}")
            return False
            
        # 파일 크기 확인
        file_size = os.path.getsize(file_path)
        logger.info(f"파일 크기: {file_size} bytes")
        
        if file_size == 0:
            logger.error("파일이 비어있습니다!")
            return False
            
        # ONNX 모델 로드 테스트
        logger.info("ONNX 모델 로드 중...")
        model = onnx.load(file_path)
        
        # 모델 검증
        logger.info("ONNX 모델 검증 중...")
        onnx.checker.check_model(model)
        
        logger.info("✅ ONNX 파일이 유효합니다!")
        logger.info(f"모델 입력: {[input.name for input in model.graph.input]}")
        logger.info(f"모델 출력: {[output.name for output in model.graph.output]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ONNX 파일 오류: {e}")
        return False

def main():
    """메인 함수"""
    onnx_file = "/runpod-volume/uvr_models/Kim_Vocal_1.onnx"
    
    logger.info("🎵 ONNX 파일 무결성 테스트 시작")
    logger.info("=" * 50)
    
    # 환경변수 확인
    env_model_dir = os.environ.get("AUDIO_SEPARATOR_MODEL_DIR")
    logger.info(f"환경변수 AUDIO_SEPARATOR_MODEL_DIR: {env_model_dir}")
    
    # 디렉토리 내용 확인
    if os.path.exists("/runpod-volume/uvr_models"):
        files = os.listdir("/runpod-volume/uvr_models")
        logger.info(f"UVR 모델 디렉토리 내용: {files}")
    else:
        logger.error("UVR 모델 디렉토리가 존재하지 않습니다!")
        
    # ONNX 파일 테스트
    success = test_onnx_file(onnx_file)
    
    if success:
        logger.info("✅ 모든 테스트 통과!")
    else:
        logger.error("❌ 테스트 실패!")
        sys.exit(1)

if __name__ == "__main__":
    main() 