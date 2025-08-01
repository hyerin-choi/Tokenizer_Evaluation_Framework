#!/usr/bin/env python3
"""
한국어 ASR 토크나이저 평가 예제 실행 스크립트
"""

import os
import sys
import subprocess
import logging

def check_dependencies():
    """필요한 패키지들이 설치되어 있는지 확인합니다."""
    required_packages = [
        'sentencepiece',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} 설치됨")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} 설치 필요")
    
    if missing_packages:
        print(f"\n다음 패키지들을 설치해주세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def run_example():
    """예제를 실행합니다."""
    print("=" * 60)
    print("한국어 ASR 토크나이저 평가 시스템")
    print("=" * 60)
    
    # 1. 의존성 확인
    print("\n1. 의존성 확인 중...")
    if not check_dependencies():
        print("\n의존성 설치 후 다시 실행해주세요.")
        return
    
    # 2. 샘플 데이터 생성
    print("\n2. 샘플 데이터 생성 중...")
    try:
        from morphological_analyzer import create_sample_data
        create_sample_data()
        print("✓ 샘플 데이터 생성 완료")
    except Exception as e:
        print(f"✗ 샘플 데이터 생성 실패: {e}")
        return
    
    # 3. 형태소 분석 실행
    print("\n3. 형태소 분석 실행 중...")
    try:
        from morphological_analyzer import MorphologicalAnalyzer
        
        with open("data/train_text.txt", 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f.readlines()]
        
        # Kiwi 분석
        try:
            kiwi_analyzer = MorphologicalAnalyzer("kiwi")
            kiwi_analyzer.save_analyzed_texts(texts, "data/train_text_morph_kiwi.txt")
            print("✓ Kiwi 형태소 분석 완료")
        except Exception as e:
            print(f"✗ Kiwi 분석 실패: {e}")
        
        # MeCab 분석
        try:
            mecab_analyzer = MorphologicalAnalyzer("mecab")
            mecab_analyzer.save_analyzed_texts(texts, "data/train_text_morph_mecab.txt")
            print("✓ MeCab 형태소 분석 완료")
        except Exception as e:
            print(f"✗ MeCab 분석 실패: {e}")
            
    except Exception as e:
        print(f"✗ 형태소 분석 실패: {e}")
        return
    
    # 4. 토크나이저 학습
    print("\n4. 토크나이저 학습 중...")
    try:
        from tokenizer_trainer import train_multiple_tokenizers
        
        config = {
            'vocab_size': 16000,
            'character_coverage': 0.9995,
            'shuffle_input_sentence': True,
            'hard_vocab_limit': False,
            'model_type': 'unigram'
        }
        
        data_config = {
            'kiwi': {'output_file': 'data/train_text_morph_kiwi.txt'},
            'mecab': {'output_file': 'data/train_text_morph_mecab.txt'},
            'output_dir': 'outputs/tokenizers'
        }
        
        model_paths = train_multiple_tokenizers(config, data_config)
        print("✓ 토크나이저 학습 완료")
        
    except Exception as e:
        print(f"✗ 토크나이저 학습 실패: {e}")
        return
    
    # 5. 평가 실행
    print("\n5. 토크나이저 평가 중...")
    try:
        from evaluator import TokenizerEvaluator
        
        evaluator = TokenizerEvaluator()
        validation_file = "data/validation_text.txt"
        output_dir = "outputs/evaluation"
        
        # Intrinsic 평가
        intrinsic_results = evaluator.evaluate_tokenizers_intrinsic(model_paths, validation_file)
        print("✓ Intrinsic 평가 완료")
        
        # Extrinsic 평가
        extrinsic_results = evaluator.evaluate_tokenizers_extrinsic(model_paths, validation_file)
        print("✓ Extrinsic 평가 완료")
        
        # 보고서 생성
        evaluator.generate_evaluation_report(intrinsic_results, extrinsic_results, output_dir)
        print("✓ 평가 보고서 생성 완료")
        
    except Exception as e:
        print(f"✗ 평가 실패: {e}")
        return
    
    # 6. 결과 요약
    print("\n" + "=" * 60)
    print("예제 실행 완료!")
    print("=" * 60)
    
    print("\n생성된 파일들:")
    if os.path.exists("outputs"):
        for root, dirs, files in os.walk("outputs"):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, ".")
                print(f"  - {relative_path}")
    
    print("\n평가 보고서:")
    report_path = "outputs/evaluation/evaluation_report.md"
    if os.path.exists(report_path):
        print(f"  - {report_path}")
        
        # 보고서 내용 일부 출력
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:20]  # 처음 20줄만 출력
                print("\n보고서 미리보기:")
                for line in lines:
                    print(line.rstrip())
                if len(lines) == 20:
                    print("...")
        except Exception as e:
            print(f"보고서 읽기 실패: {e}")
    
    print("\n" + "=" * 60)
    print("다음 명령어로 전체 파이프라인을 실행할 수 있습니다:")
    print("python main.py")
    print("=" * 60)

if __name__ == "__main__":
    run_example() 