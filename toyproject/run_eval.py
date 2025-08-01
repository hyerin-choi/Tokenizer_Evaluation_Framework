#!/usr/bin/env python3
"""
한국어 ASR 토크나이저 평가 스크립트
"""

import os
import logging
import yaml
import time
import argparse

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('evaluation.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def check_files():
    """필요한 파일들 확인"""
    print("=== 파일 확인 ===")
    
    # 설정 파일
    if not os.path.exists("config.yaml"):
        print("✗ config.yaml 파일이 없습니다")
        return False
    print("✓ config.yaml 확인")
    
    # 모델 파일들
    model_files = {
        "kiwi": "outputs/tokenizers/tokenizer_kiwi.model",
        "mecab": "outputs/tokenizers/tokenizer_mecab.model"
    }
    
    missing_models = []
    for name, path in model_files.items():
        if os.path.exists(path):
            print(f"✓ {name} 모델 확인: {path}")
        else:
            print(f"✗ {name} 모델 없음: {path}")
            missing_models.append(name)
    
    if missing_models:
        print(f"\n⚠️  다음 모델들이 없습니다: {missing_models}")
        print("먼저 토크나이저 학습을 실행해주세요: python main.py")
        return False
    
    # 데이터 파일 확인
    try:
        with open("config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        validation_file = config['data']['validation_file']
        if os.path.exists(validation_file):
            print(f"✓ 검증 데이터 확인: {validation_file}")
        else:
            print(f"✗ 검증 데이터 없음: {validation_file}")
            return False
            
    except Exception as e:
        print(f"✗ 설정 파일 읽기 실패: {e}")
        return False
    
    return True

def run_evaluation():
    """평가 실행"""
    print("\n=== 토크나이저 평가 시작 ===")
    
    try:
        # 설정 로드
        with open("config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 모델 경로
        model_paths = {
            "kiwi": "outputs/tokenizers/tokenizer_kiwi.model",
            "mecab": "outputs/tokenizers/tokenizer_mecab.model"
        }
        
        validation_file = config['data']['validation_file']
        output_dir = os.path.join(config['data']['output_dir'], 'evaluation')
        
        # 평가 실행
        from evaluator import TokenizerEvaluator
        evaluator = TokenizerEvaluator()
        
        # Intrinsic 평가
        print("Intrinsic 평가 중...")
        intrinsic_results = evaluator.evaluate_tokenizers_intrinsic(model_paths, validation_file)
        print(f"✓ Intrinsic 평가 완료: {len(intrinsic_results)}개 토크나이저")
        
        # Extrinsic 평가
        print("Extrinsic 평가 중...")
        extrinsic_results = evaluator.evaluate_tokenizers_extrinsic(model_paths, validation_file)
        print(f"✓ Extrinsic 평가 완료: {len(extrinsic_results)}개 토크나이저")
        
        # 보고서 생성
        print("평가 보고서 생성 중...")
        evaluator.generate_evaluation_report(intrinsic_results, extrinsic_results, output_dir)
        print("✓ 평가 보고서 생성 완료")
        
        return True
        
    except Exception as e:
        print(f"✗ 평가 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_results():
    """결과 표시"""
    print("\n=== 평가 결과 ===")
    
    # CSV 결과 표시
    csv_files = [
        ("Intrinsic", "outputs/evaluation/intrinsic_results.csv"),
        ("Extrinsic", "outputs/evaluation/extrinsic_results.csv")
    ]
    
    for name, path in csv_files:
        if os.path.exists(path):
            print(f"\n{name} 평가 결과:")
            try:
                import pandas as pd
                df = pd.read_csv(path)
                print(df.to_string(index=False))
            except Exception as e:
                print(f"CSV 읽기 실패: {e}")
        else:
            print(f"✗ {name} 결과 파일 없음: {path}")
    
    # 보고서 확인
    report_path = "outputs/evaluation/evaluation_report.md"
    if os.path.exists(report_path):
        print(f"\n✓ 평가 보고서: {report_path}")
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print("\n" + "="*50)
                print("평가 보고서:")
                print("="*50)
                print(content)
                print("="*50)
        except Exception as e:
            print(f"보고서 읽기 실패: {e}")
    else:
        print(f"✗ 평가 보고서 없음: {report_path}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='한국어 ASR 토크나이저 평가')
    parser.add_argument('--check-only', action='store_true', help='파일 확인만')
    parser.add_argument('--full', action='store_true', help='전체 파이프라인 실행')
    
    args = parser.parse_args()
    
    setup_logging()
    start_time = time.time()
    
    try:
        if args.check_only:
            # 파일 확인만
            if check_files():
                print("\n✓ 모든 파일이 준비되었습니다")
            else:
                print("\n✗ 파일 확인 실패")
            return
        
        if args.full:
            # 전체 파이프라인 실행
            print("=== 전체 파이프라인 실행 ===")
            from main import main as run_main
            run_main()
        else:
            # 평가만 실행
            if not check_files():
                print("\n✗ 파일 확인 실패")
                return
            
            if run_evaluation():
                total_time = time.time() - start_time
                print(f"\n✓ 평가 완료 (소요시간: {total_time:.2f}초)")
                show_results()
                print("\n" + "="*50)
                print("🎉 한국어 ASR 토크나이저 평가 완료!")
                print("="*50)
            else:
                print("\n✗ 평가 실패")
                
    except Exception as e:
        print(f"✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 