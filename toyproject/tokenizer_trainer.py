"""
SentencePiece Unigram 토크나이저 학습 모듈
"""

import os
import logging
import subprocess
import tempfile
from typing import Dict, Any, List
import json
import time

class SentencePieceTrainer:
    """SentencePiece 토크나이저 학습 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 토크나이저 설정 딕셔너리
        """
        self.config = config
        self.model_prefix = None
        self.vocab_size = config.get('vocab_size', 16000)
        self.character_coverage = config.get('character_coverage', 0.9995)
        self.shuffle_input_sentence = config.get('shuffle_input_sentence', True)
        self.hard_vocab_limit = config.get('hard_vocab_limit', False)
        self.model_type = config.get('model_type', 'unigram')
        
    def train_tokenizer(self, input_file: str, output_dir: str, model_name: str) -> str:
        """
        SentencePiece 토크나이저를 학습합니다.
        
        Args:
            input_file: 학습 데이터 파일 경로
            output_dir: 출력 디렉토리
            model_name: 모델 이름
            
        Returns:
            str: 학습된 모델 파일 경로
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 모델 파일 경로 설정
        model_path = os.path.join(output_dir, f"{model_name}.model")
        vocab_path = os.path.join(output_dir, f"{model_name}.vocab")
        
        # SentencePiece 학습 명령어 구성
        cmd = [
            "spm_train",
            f"--input={input_file}",
            f"--model_prefix={os.path.join(output_dir, model_name)}",
            f"--vocab_size={self.vocab_size}",
            f"--character_coverage={self.character_coverage}",
            f"--model_type={self.model_type}",
            "--input_sentence_size=10000000",  # 메모리 사용량 제한
            "--shuffle_input_sentence=true" if self.shuffle_input_sentence else "--shuffle_input_sentence=false",
            "--hard_vocab_limit=false" if not self.hard_vocab_limit else "--hard_vocab_limit=true",
            "--num_threads=8",  # 병렬 처리
            "--train_extremely_large_corpus=true"  # 대용량 코퍼스 지원
        ]
        
        logging.info(f"토크나이저 학습 시작: {model_name}")
        logging.info(f"입력 파일: {input_file}")
        logging.info(f"출력 디렉토리: {output_dir}")
        
        start_time = time.time()
        
        try:
            # SentencePiece 학습 실행
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            training_time = time.time() - start_time
            logging.info(f"토크나이저 학습 완료: {model_name} (소요시간: {training_time:.2f}초)")
            
            # 학습 결과 확인
            if os.path.exists(model_path):
                logging.info(f"모델 파일 생성됨: {model_path}")
                return model_path
            else:
                raise FileNotFoundError(f"모델 파일이 생성되지 않았습니다: {model_path}")
                
        except subprocess.CalledProcessError as e:
            logging.error(f"토크나이저 학습 실패: {e}")
            logging.error(f"stdout: {e.stdout}")
            logging.error(f"stderr: {e.stderr}")
            raise
        except Exception as e:
            logging.error(f"토크나이저 학습 중 오류 발생: {e}")
            raise
    
    def encode_text(self, model_path: str, text: str) -> List[str]:
        """
        텍스트를 토큰화합니다.
        
        Args:
            model_path: 학습된 모델 파일 경로
            text: 토큰화할 텍스트
            
        Returns:
            List[str]: 토큰 리스트
        """
        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.load(model_path)
            return sp.encode_as_pieces(text)
        except ImportError:
            logging.error("sentencepiece가 설치되지 않았습니다.")
            raise
    
    def get_vocab_info(self, model_path: str) -> Dict[str, Any]:
        """
        어휘 정보를 가져옵니다.
        
        Args:
            model_path: 모델 파일 경로
            
        Returns:
            Dict[str, Any]: 어휘 정보
        """
        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.load(model_path)
            
            vocab_size = sp.get_piece_size()
            vocab_list = [sp.id_to_piece(i) for i in range(vocab_size)]
            
            return {
                "vocab_size": vocab_size,
                "vocab_list": vocab_list,
                "model_path": model_path
            }
        except ImportError:
            logging.error("sentencepiece가 설치되지 않았습니다.")
            raise

def train_multiple_tokenizers(config: Dict[str, Any], data_config: Dict[str, Any]):
    """
    여러 토크나이저를 학습합니다.
    
    Args:
        config: 토크나이저 설정
        data_config: 데이터 설정
    """
    trainer = SentencePieceTrainer(config)
    
    # Kiwi 기반 토크나이저 학습
    if os.path.exists(data_config['kiwi']['output_file']):
        logging.info("Kiwi 기반 토크나이저 학습 시작...")
        kiwi_model_path = trainer.train_tokenizer(
            input_file=data_config['kiwi']['output_file'],
            output_dir=data_config['output_dir'],
            model_name="tokenizer_kiwi"
        )
        logging.info(f"Kiwi 토크나이저 학습 완료: {kiwi_model_path}")
    else:
        logging.warning(f"Kiwi 형태소 분석 파일이 없습니다: {data_config['kiwi']['output_file']}")
        kiwi_model_path = None
    
    # MeCab 기반 토크나이저 학습
    if os.path.exists(data_config['mecab']['output_file']):
        logging.info("MeCab 기반 토크나이저 학습 시작...")
        mecab_model_path = trainer.train_tokenizer(
            input_file=data_config['mecab']['output_file'],
            output_dir=data_config['output_dir'],
            model_name="tokenizer_mecab"
        )
        logging.info(f"MeCab 토크나이저 학습 완료: {mecab_model_path}")
    else:
        logging.warning(f"MeCab 형태소 분석 파일이 없습니다: {data_config['mecab']['output_file']}")
        mecab_model_path = None
    
    return {
        "kiwi": kiwi_model_path,
        "mecab": mecab_model_path
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 설정 예시
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
    
    # 토크나이저 학습
    model_paths = train_multiple_tokenizers(config, data_config)
    
    print("토크나이저 학습이 완료되었습니다.")
    print(f"학습된 모델들: {model_paths}") 