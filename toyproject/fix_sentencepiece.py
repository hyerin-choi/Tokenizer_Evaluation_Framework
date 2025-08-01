#!/usr/bin/env python3
"""
sentencepiece 문제 해결 스크립트
"""

import subprocess
import sys

def fix_sentencepiece():
    """sentencepiece 문제를 해결합니다."""
    print("=== sentencepiece 문제 해결 ===")
    
    # 1. 현재 설치된 sentencepiece 확인
    print("1. 현재 sentencepiece 상태 확인...")
    try:
        import sentencepiece as spm
        print(f"✓ sentencepiece 설치됨")
        print(f"  버전: {spm.__version__}")
        print(f"  사용 가능한 속성들: {[attr for attr in dir(spm) if not attr.startswith('_')]}")
    except ImportError:
        print("✗ sentencepiece가 설치되지 않음")
    except Exception as e:
        print(f"✗ sentencepiece 오류: {e}")
    
    # 2. sentencepiece 재설치
    print("\n2. sentencepiece 재설치...")
    try:
        # 기존 제거
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "sentencepiece"], 
                      check=True, capture_output=True)
        print("✓ 기존 sentencepiece 제거 완료")
        
        # 새로 설치
        subprocess.run([sys.executable, "-m", "pip", "install", "sentencepiece==0.1.99"], 
                      check=True, capture_output=True)
        print("✓ sentencepiece 0.1.99 설치 완료")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ 설치 실패: {e}")
        return False
    
    # 3. 설치 확인
    print("\n3. 설치 확인...")
    try:
        import sentencepiece as spm
        print(f"✓ sentencepiece 재설치 성공")
        print(f"  버전: {spm.__version__}")
        
        # SentencePieceProcessor 확인
        if hasattr(spm, 'SentencePieceProcessor'):
            print("✓ SentencePieceProcessor 사용 가능")
        else:
            print("✗ SentencePieceProcessor를 찾을 수 없음")
            # 대안 방법 시도
            if hasattr(spm, 'SentencePieceProcessor'):
                print("✓ 대안 방법으로 SentencePieceProcessor 사용 가능")
            else:
                print("✗ 모든 방법 실패")
                return False
        
        # 테스트
        print("\n4. sentencepiece 테스트...")
        sp = spm.SentencePieceProcessor()
        print("✓ SentencePieceProcessor 초기화 성공")
        
        return True
        
    except Exception as e:
        print(f"✗ 확인 실패: {e}")
        return False

def test_evaluator():
    """evaluator 모듈 테스트"""
    print("\n=== evaluator 모듈 테스트 ===")
    
    try:
        from evaluator import TokenizerEvaluator
        evaluator = TokenizerEvaluator()
        print("✓ TokenizerEvaluator 초기화 성공")
        
        # sentencepiece 테스트
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        print("✓ sentencepiece 테스트 성공")
        
        return True
        
    except Exception as e:
        print(f"✗ evaluator 테스트 실패: {e}")
        return False

def main():
    """메인 함수"""
    print("sentencepiece 문제 해결 스크립트")
    print("=" * 50)
    
    # 1. sentencepiece 수정
    if fix_sentencepiece():
        print("\n✓ sentencepiece 문제 해결 완료")
    else:
        print("\n✗ sentencepiece 문제 해결 실패")
        return
    
    # 2. evaluator 테스트
    if test_evaluator():
        print("\n✓ evaluator 모듈 테스트 성공")
    else:
        print("\n✗ evaluator 모듈 테스트 실패")
        return
    
    print("\n" + "=" * 50)
    print("이제 다음 명령어로 평가를 실행할 수 있습니다:")
    print("python main.py --skip-morph --skip-training")
    print("=" * 50)

if __name__ == "__main__":
    main() 