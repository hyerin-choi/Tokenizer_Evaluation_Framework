"""
한국어 ASR 토크나이저 평가 메인 스크립트
전체 파이프라인을 실행합니다.
"""

import os
import logging
import yaml
import argparse
from typing import Dict, Any
import time

from morphological_analyzer import MorphologicalAnalyzer, create_sample_data
from tokenizer_trainer import train_multiple_tokenizers
from evaluator import TokenizerEvaluator

def load_config(config_path: str) -> Dict[str, Any]:
    """설정 파일을 로드합니다."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(log_level: str = "INFO"):
    """로깅을 설정합니다."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('tokenizer_evaluation.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def run_morphological_analysis(config: Dict[str, Any]):
    """형태소 분석을 실행합니다."""
    logging.info("=== 형태소 분석 시작 ===")
    
    # 학습 데이터 로드
    train_file = config['data']['train_file']
    if not os.path.exists(train_file):
        logging.warning(f"학습 데이터 파일이 없습니다: {train_file}")
        logging.info("샘플 데이터를 생성합니다...")
        create_sample_data()
    
    with open(train_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    
    # Kiwi 형태소 분석
    if config['morphological_analysis']['kiwi']['enabled']:
        logging.info("Kiwi 형태소 분석 시작...")
        kiwi_analyzer = MorphologicalAnalyzer("kiwi")
        kiwi_analyzer.save_analyzed_texts(
            texts, 
            config['morphological_analysis']['kiwi']['output_file']
        )
    
    # MeCab 형태소 분석
    if config['morphological_analysis']['mecab']['enabled']:
        logging.info("MeCab 형태소 분석 시작...")
        mecab_analyzer = MorphologicalAnalyzer("mecab")
        mecab_analyzer.save_analyzed_texts(
            texts, 
            config['morphological_analysis']['mecab']['output_file']
        )
    
    logging.info("형태소 분석 완료")

def run_tokenizer_training(config: Dict[str, Any]) -> Dict[str, str]:
    """토크나이저 학습을 실행합니다."""
    logging.info("=== 토크나이저 학습 시작 ===")
    
    # 데이터 설정
    data_config = {
        'kiwi': {'output_file': config['morphological_analysis']['kiwi']['output_file']},
        'mecab': {'output_file': config['morphological_analysis']['mecab']['output_file']},
        'output_dir': os.path.join(config['data']['output_dir'], 'tokenizers')
    }
    
    # 토크나이저 학습
    model_paths = train_multiple_tokenizers(config['tokenizer'], data_config)
    
    logging.info("토크나이저 학습 완료")
    return model_paths

def run_evaluation(config: Dict[str, Any], model_paths: Dict[str, str]):
    """토크나이저 평가를 실행합니다."""
    logging.info("=== 토크나이저 평가 시작 ===")
    
    evaluator = TokenizerEvaluator()
    validation_file = config['data']['validation_file']
    output_dir = os.path.join(config['data']['output_dir'], 'evaluation')
    
    # Intrinsic 평가
    if config['evaluation']['intrinsic']['enabled']:
        logging.info("Intrinsic 평가 시작...")
        intrinsic_results = evaluator.evaluate_tokenizers_intrinsic(model_paths, validation_file)
    else:
        intrinsic_results = {}
    
    # Extrinsic 평가
    if config['evaluation']['extrinsic']['enabled']:
        logging.info("Extrinsic 평가 시작...")
        extrinsic_results = evaluator.evaluate_tokenizers_extrinsic(model_paths, validation_file)
    else:
        extrinsic_results = {}
    
    # 보고서 생성
    if intrinsic_results or extrinsic_results:
        evaluator.generate_evaluation_report(intrinsic_results, extrinsic_results, output_dir)
    
    logging.info("토크나이저 평가 완료")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="한국어 ASR 토크나이저 평가")
    parser.add_argument("--config", default="config.yaml", help="설정 파일 경로")
    parser.add_argument("--log-level", default="INFO", help="로그 레벨")
    parser.add_argument("--skip-morph", action="store_true", help="형태소 분석 건너뛰기")
    parser.add_argument("--skip-training", action="store_true", help="토크나이저 학습 건너뛰기")
    parser.add_argument("--skip-evaluation", action="store_true", help="평가 건너뛰기")
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging(args.log_level)
    
    # 설정 로드
    config = load_config(args.config)
    
    # 출력 디렉토리 생성
    os.makedirs(config['data']['output_dir'], exist_ok=True)
    
    start_time = time.time()
    
    try:
        # 1. 형태소 분석
        if not args.skip_morph:
            run_morphological_analysis(config)
        
        # 2. 토크나이저 학습
        if not args.skip_training:
            model_paths = run_tokenizer_training(config)
        else:
            # 기존 모델 경로 사용
            model_paths = {
                "kiwi": os.path.join(config['data']['output_dir'], 'tokenizers', 'tokenizer_kiwi.model'),
                "mecab": os.path.join(config['data']['output_dir'], 'tokenizers', 'tokenizer_mecab.model')
            }
        
        # 3. 토크나이저 평가
        if not args.skip_evaluation:
            run_evaluation(config, model_paths)
        
        total_time = time.time() - start_time
        logging.info(f"전체 파이프라인 완료 (소요시간: {total_time:.2f}초)")
        
        # 결과 요약
        print("\n" + "="*50)
        print("한국어 ASR 토크나이저 평가 완료")
        print("="*50)
        print(f"출력 디렉토리: {config['data']['output_dir']}")
        print(f"소요시간: {total_time:.2f}초")
        print("\n생성된 파일들:")
        
        # 생성된 파일 목록 출력
        for root, dirs, files in os.walk(config['data']['output_dir']):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, config['data']['output_dir'])
                print(f"  - {relative_path}")
        
        print("\n평가 보고서:")
        report_path = os.path.join(config['data']['output_dir'], 'evaluation', 'evaluation_report.md')
        if os.path.exists(report_path):
            print(f"  - {report_path}")
        
        print("\n" + "="*50)
        
    except Exception as e:
        logging.error(f"파이프라인 실행 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main() 