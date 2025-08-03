#!/usr/bin/env python3
"""
ONNX 모델 파일 버전 업그레이드 스크립트
"""

import os
import onnx
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upgrade_onnx_model(input_path, output_path):
    """ONNX 모델 버전 업그레이드"""
    try:
        logger.info(f"업그레이드 중: {input_path}")
        
        # ONNX 모델 로드
        model = onnx.load(input_path)
        
        # 현재 IR 버전 확인
        current_version = model.ir_version
        logger.info(f"현재 IR 버전: {current_version}")
        
        if current_version < 3:
            logger.info("IR 버전이 3 미만입니다. 업그레이드를 진행합니다.")
            
            # IR 버전을 최신으로 업그레이드
            model.ir_version = 8  # 최신 안정 버전
            model.producer_name = "AICoverGen"
            model.producer_version = "1.0"
            
            # 모델 저장
            onnx.save(model, output_path)
            
            # 검증
            onnx.checker.check_model(output_path)
            
            logger.info(f"✅ 업그레이드 완료: {output_path}")
            return True
        else:
            logger.info(f"이미 최신 버전입니다 (IR 버전: {current_version})")
            return True
            
    except Exception as e:
        logger.error(f"❌ 업그레이드 실패: {e}")
        return False

def main():
    """메인 함수"""
    uvr_models_dir = "/runpod-volume/uvr_models"
    
    logger.info("🎵 ONNX 모델 버전 업그레이드 시작")
    logger.info("=" * 50)
    
    if not os.path.exists(uvr_models_dir):
        logger.error(f"UVR 모델 디렉토리가 존재하지 않습니다: {uvr_models_dir}")
        return
    
    # ONNX 파일들 찾기
    onnx_files = []
    for file in os.listdir(uvr_models_dir):
        if file.endswith('.onnx'):
            onnx_files.append(file)
    
    logger.info(f"발견된 ONNX 파일들: {onnx_files}")
    
    if not onnx_files:
        logger.warning("업그레이드할 ONNX 파일이 없습니다.")
        return
    
    # 각 파일 업그레이드
    success_count = 0
    for onnx_file in onnx_files:
        input_path = os.path.join(uvr_models_dir, onnx_file)
        output_path = os.path.join(uvr_models_dir, f"upgraded_{onnx_file}")
        
        if upgrade_onnx_model(input_path, output_path):
            # 원본 파일 백업하고 업그레이드된 파일로 교체
            backup_path = os.path.join(uvr_models_dir, f"backup_{onnx_file}")
            os.rename(input_path, backup_path)
            os.rename(output_path, input_path)
            logger.info(f"✅ {onnx_file} 업그레이드 및 교체 완료")
            success_count += 1
    
    logger.info(f"✅ 완료: {success_count}/{len(onnx_files)} 파일 업그레이드 성공")
    
    # 최종 파일 목록 확인
    final_files = os.listdir(uvr_models_dir)
    logger.info(f"최종 파일들: {final_files}")

if __name__ == "__main__":
    main() 