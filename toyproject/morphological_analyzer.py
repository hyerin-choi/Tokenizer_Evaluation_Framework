"""
한국어 형태소 분석기 모듈
Kiwi와 MeCab-ko 기반 형태소 분석을 지원합니다.
"""

import os
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

try:
    from kiwipiepy import Kiwi
except ImportError:
    Kiwi = None
    logging.warning("Kiwi가 설치되지 않았습니다. pip install kiwipiepy로 설치하세요.")

try:
    from konlpy.tag import Mecab
except ImportError:
    Mecab = None
    logging.warning("MeCab이 설치되지 않았습니다. pip install konlpy로 설치하세요.")

@dataclass
class MorphologicalResult:
    """형태소 분석 결과를 저장하는 데이터 클래스"""
    original_text: str
    analyzed_text: str
    tokens: List[str]
    pos_tags: List[str]
    analyzer_name: str

class MorphologicalAnalyzer:
    """한국어 형태소 분석기 클래스"""
    
    def __init__(self, analyzer_type: str = "kiwi"):
        """
        Args:
            analyzer_type: "kiwi" 또는 "mecab"
        """
        self.analyzer_type = analyzer_type
        self.kiwi = None
        self.mecab = None
        
        if analyzer_type == "kiwi":
            if Kiwi is not None:
                self.kiwi = Kiwi()
            else:
                raise ImportError("Kiwi가 설치되지 않았습니다.")
        elif analyzer_type == "mecab":
            if Mecab is not None:
                self.mecab = Mecab()
            else:
                raise ImportError("MeCab이 설치되지 않았습니다.")
        else:
            raise ValueError("analyzer_type은 'kiwi' 또는 'mecab'이어야 합니다.")
    
    def analyze_text(self, text: str) -> MorphologicalResult:
        """
        텍스트를 형태소 분석합니다.
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            MorphologicalResult: 형태소 분석 결과
        """
        if self.analyzer_type == "kiwi":
            return self._analyze_with_kiwi(text)
        elif self.analyzer_type == "mecab":
            return self._analyze_with_mecab(text)
    
    def _analyze_with_kiwi(self, text: str) -> MorphologicalResult:
        """Kiwi를 사용한 형태소 분석"""
        tokens = []
        pos_tags = []
        
        # Kiwi 형태소 분석
        result = self.kiwi.analyze(text)
        
        for token, pos, start, end in result[0][0]:
            tokens.append(token)
            pos_tags.append(pos)
        
        # 분석된 텍스트 생성 (형태소 + 공백)
        analyzed_text = " ".join(tokens)
        
        return MorphologicalResult(
            original_text=text,
            analyzed_text=analyzed_text,
            tokens=tokens,
            pos_tags=pos_tags,
            analyzer_name="kiwi"
        )
    
    def _analyze_with_mecab(self, text: str) -> MorphologicalResult:
        """MeCab을 사용한 형태소 분석"""
        tokens = []
        pos_tags = []
        
        # MeCab 형태소 분석
        pos_result = self.mecab.pos(text)
        
        for token, pos in pos_result:
            tokens.append(token)
            pos_tags.append(pos)
        
        # 분석된 텍스트 생성 (형태소 + 공백)
        analyzed_text = " ".join(tokens)
        
        return MorphologicalResult(
            original_text=text,
            analyzed_text=analyzed_text,
            tokens=tokens,
            pos_tags=pos_tags,
            analyzer_name="mecab"
        )
    
    def batch_analyze(self, texts: List[str]) -> List[MorphologicalResult]:
        """
        여러 텍스트를 배치로 형태소 분석합니다.
        
        Args:
            texts: 분석할 텍스트 리스트
            
        Returns:
            List[MorphologicalResult]: 형태소 분석 결과 리스트
        """
        results = []
        for text in texts:
            try:
                result = self.analyze_text(text)
                results.append(result)
            except Exception as e:
                logging.error(f"텍스트 분석 중 오류 발생: {text[:50]}... - {e}")
                # 오류 발생 시 원본 텍스트를 그대로 사용
                results.append(MorphologicalResult(
                    original_text=text,
                    analyzed_text=text,
                    tokens=[text],
                    pos_tags=["UNK"],
                    analyzer_name=self.analyzer_type
                ))
        return results
    
    def save_analyzed_texts(self, texts: List[str], output_file: str):
        """
        형태소 분석 결과를 파일로 저장합니다.
        
        Args:
            texts: 분석할 텍스트 리스트
            output_file: 출력 파일 경로
        """
        results = self.batch_analyze(texts)
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(result.analyzed_text + '\n')
        
        logging.info(f"{len(results)}개 텍스트의 형태소 분석 결과를 {output_file}에 저장했습니다.")

def create_sample_data():
    """샘플 데이터를 생성합니다."""
    sample_texts = [
        "안녕하세요 고객님 무엇을 도와드릴까요",
        "웹사이트 접속이 안 되는 문제가 있습니다",
        "SITE 플랫폼에서 CORD 서비스를 이용하고 있어요",
        "상담사 연결을 원하시나요",
        "외래어와 방송 플랫폼 단어들이 많이 나오네요",
        "실시간 ASR 시스템이 잘 작동하고 있습니다",
        "음성 인식 정확도가 높아졌어요",
        "토크나이저 성능 평가를 진행하고 있습니다"
    ]
    
    os.makedirs("data", exist_ok=True)
    
    with open("data/train_text.txt", 'w', encoding='utf-8') as f:
        for text in sample_texts:
            f.write(text + '\n')
    
    with open("data/validation_text.txt", 'w', encoding='utf-8') as f:
        for text in sample_texts[:4]:  # 검증용으로 일부만 사용
            f.write(text + '\n')
    
    logging.info("샘플 데이터를 생성했습니다.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 샘플 데이터 생성
    create_sample_data()
    
    # 형태소 분석 테스트
    with open("data/train_text.txt", 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    
    # Kiwi 분석
    kiwi_analyzer = MorphologicalAnalyzer("kiwi")
    kiwi_analyzer.save_analyzed_texts(texts, "data/train_text_morph_kiwi.txt")
    
    # MeCab 분석
    mecab_analyzer = MorphologicalAnalyzer("mecab")
    mecab_analyzer.save_analyzed_texts(texts, "data/train_text_morph_mecab.txt")
    
    print("형태소 분석이 완료되었습니다.") 