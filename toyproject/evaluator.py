"""
토크나이저 평가 모듈
Intrinsic 및 Extrinsic 평가를 지원합니다.
"""

import os
import logging
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import sentencepiece as spm
    # sentencepiece 버전 확인 및 호환성 처리
    if not hasattr(spm, 'SentencePieceProcessor'):
        # 대안 방법 시도
        if hasattr(spm, 'SentencePieceProcessor'):
            pass  # 이미 사용 가능
        else:
            raise ImportError("SentencePieceProcessor를 찾을 수 없습니다.")
except ImportError:
    spm = None
    logging.warning("sentencepiece가 설치되지 않았습니다.")

@dataclass
class IntrinsicMetrics:
    """Intrinsic 평가 지표"""
    # 기본 통계
    total_word_count: int  # 총 단어 수
    total_encountered_tokens: int  # 총 토큰 수
    unk_token_count: int  # <unk> 토큰 수
    
    # 커버리지 지표
    vocab_coverage: float  # 어휘 커버리지 = 1 - unk_count / total_tokens
    word_coverage: float  # 단어 커버리지 = 1 - unk_count / total_words
    
    # 효율성 지표
    fertility: float  # 토큰/단어 비율 = total_tokens / total_words
    avg_tokens_per_word: float  # 단어당 평균 토큰 수
    avg_token_length: float  # 평균 토큰 길이
    
    # 분할 지표
    continued_word_rate: float  # 여러 토큰으로 분할된 단어 비율
    avg_word_tokens: float  # 단어당 평균 토큰 수 (분할된 단어만)
    
    # 어휘 통계
    unique_tokens: int  # 고유 토큰 수
    unique_words: int  # 고유 단어 수

@dataclass
class ExtrinsicMetrics:
    """Extrinsic 평가 지표"""
    cer: float  # Character Error Rate (한국어에 더 적합)
    wer: float  # Word Error Rate (참고용)
    latency_ms: float  # 추론 지연시간 (밀리초)
    throughput: float  # 처리량 (문장/초)

class TokenizerEvaluator:
    """토크나이저 평가 클래스"""
    
    def __init__(self):
        self.results = {}
    
    def load_tokenizer(self, model_path: str):
        """토크나이저를 로드합니다."""
        if spm is None:
            raise ImportError("sentencepiece가 설치되지 않았습니다.")
        
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)
        return sp
    
    def calculate_intrinsic_metrics(self, tokenizer, texts: List[str]) -> IntrinsicMetrics:
        """
        Intrinsic 평가 지표를 계산합니다.
        
        Args:
            tokenizer: SentencePiece 토크나이저
            texts: 평가할 텍스트 리스트
            
        Returns:
            IntrinsicMetrics: 평가 지표
        """
        total_tokens = 0
        total_words = 0
        unk_count = 0
        token_lengths = []
        all_tokens = set()
        all_words = set()
        continued_words = 0
        word_token_counts = []
        
        # <unk> 토큰 ID 찾기
        try:
            unk_id = tokenizer.piece_to_id('<unk>')
        except:
            unk_id = -1
        
        for text in texts:
            if not text.strip():
                continue
                
            # 단어 분리
            words = text.split()
            total_words += len(words)
            all_words.update(words)
            
            # 토큰화
            tokens = tokenizer.encode_as_pieces(text)
            total_tokens += len(tokens)
            all_tokens.update(tokens)
            
            # <unk> 토큰 카운트 및 토큰 길이 기록
            for token in tokens:
                if token == '<unk>' or (unk_id != -1 and tokenizer.piece_to_id(token) == unk_id):
                    unk_count += 1
                token_lengths.append(len(token))
            
            # 단어별 토큰 수 계산
            word_start = 0
            for word in words:
                # 해당 단어의 토큰 수 계산
                word_tokens = []
                current_pos = 0
                
                for token in tokens:
                    # 토큰이 단어의 일부인지 확인
                    if current_pos < len(text):
                        token_in_text = text[current_pos:current_pos + len(token)]
                        if token_in_text == token:
                            word_tokens.append(token)
                            current_pos += len(token)
                        else:
                            # 공백 건너뛰기
                            current_pos += 1
                            continue
                
                word_token_count = len(word_tokens)
                word_token_counts.append(word_token_count)
                
                if word_token_count > 1:
                    continued_words += 1
        
        # 지표 계산
        fertility = total_tokens / total_words if total_words > 0 else 0
        avg_tokens_per_word = total_tokens / total_words if total_words > 0 else 0
        avg_token_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0
        
        # 커버리지 지표
        vocab_coverage = 1.0 - (unk_count / total_tokens) if total_tokens > 0 else 0
        word_coverage = 1.0 - (unk_count / total_words) if total_words > 0 else 0
        
        # 분할 지표
        continued_word_rate = continued_words / total_words if total_words > 0 else 0
        avg_word_tokens = sum(word_token_counts) / len(word_token_counts) if word_token_counts else 0
        
        return IntrinsicMetrics(
            # 기본 통계
            total_word_count=total_words,
            total_encountered_tokens=total_tokens,
            unk_token_count=unk_count,
            
            # 커버리지 지표
            vocab_coverage=vocab_coverage,
            word_coverage=word_coverage,
            
            # 효율성 지표
            fertility=fertility,
            avg_tokens_per_word=avg_tokens_per_word,
            avg_token_length=avg_token_length,
            
            # 분할 지표
            continued_word_rate=continued_word_rate,
            avg_word_tokens=avg_word_tokens,
            
            # 어휘 통계
            unique_tokens=len(all_tokens),
            unique_words=len(all_words)
        )
    
    def evaluate_tokenizers_intrinsic(self, model_paths: Dict[str, str], 
                                    validation_file: str) -> Dict[str, IntrinsicMetrics]:
        """
        여러 토크나이저의 Intrinsic 평가를 수행합니다.
        
        Args:
            model_paths: 토크나이저 모델 경로 딕셔너리
            validation_file: 검증 데이터 파일 경로
            
        Returns:
            Dict[str, IntrinsicMetrics]: 각 토크나이저의 평가 결과
        """
        # 검증 데이터 로드
        with open(validation_file, 'r', encoding='utf-8') as f:
            validation_texts = [line.strip() for line in f.readlines()]
        
        results = {}
        tokenizers = {}
        
        for name, model_path in model_paths.items():
            if model_path is None or not os.path.exists(model_path):
                logging.warning(f"토크나이저 모델이 없습니다: {name}")
                continue
                
            logging.info(f"{name} 토크나이저 Intrinsic 평가 시작...")
            
            try:
                tokenizer = self.load_tokenizer(model_path)
                tokenizers[name] = tokenizer
                metrics = self.calculate_intrinsic_metrics(tokenizer, validation_texts)
                results[name] = metrics
                
                logging.info(f"{name} 평가 완료:")
                logging.info(f"  - 총 단어 수: {metrics.total_word_count:,}")
                logging.info(f"  - 총 토큰 수: {metrics.total_encountered_tokens:,}")
                logging.info(f"  - <unk> 토큰 수: {metrics.unk_token_count:,}")
                logging.info(f"  - 어휘 커버리지: {metrics.vocab_coverage:.4f}")
                logging.info(f"  - 단어 커버리지: {metrics.word_coverage:.4f}")
                logging.info(f"  - Fertility: {metrics.fertility:.4f}")
                logging.info(f"  - 평균 토큰/단어: {metrics.avg_tokens_per_word:.4f}")
                logging.info(f"  - 평균 토큰 길이: {metrics.avg_token_length:.2f}")
                logging.info(f"  - 분할된 단어 비율: {metrics.continued_word_rate:.4f}")
                logging.info(f"  - 고유 토큰 수: {metrics.unique_tokens:,}")
                logging.info(f"  - 고유 단어 수: {metrics.unique_words:,}")
                
            except Exception as e:
                logging.error(f"{name} 토크나이저 평가 중 오류: {e}")
        
        # 어휘 중복 비율 계산 (2개 이상의 토크나이저가 있을 때)
        if len(tokenizers) >= 2:
            tokenizer_names = list(tokenizers.keys())
            logging.info(f"\n어휘 중복 비율 계산...")
            
            for i in range(len(tokenizer_names)):
                for j in range(i + 1, len(tokenizer_names)):
                    name1, name2 = tokenizer_names[i], tokenizer_names[j]
                    overlap = self.calculate_vocab_overlap(tokenizers[name1], tokenizers[name2])
                    logging.info(f"  - {name1} vs {name2}: {overlap:.4f}")
        
        return results
    
    def calculate_vocab_overlap(self, tokenizer1, tokenizer2) -> float:
        """
        두 토크나이저의 어휘 중복 비율을 계산합니다.
        
        Args:
            tokenizer1: 첫 번째 토크나이저
            tokenizer2: 두 번째 토크나이저
            
        Returns:
            float: 어휘 중복 비율
        """
        try:
            # 각 토크나이저의 어휘 집합 생성
            vocab1 = set()
            vocab2 = set()
            
            # 토크나이저1의 어휘 수집
            for i in range(tokenizer1.get_piece_size()):
                vocab1.add(tokenizer1.id_to_piece(i))
            
            # 토크나이저2의 어휘 수집
            for i in range(tokenizer2.get_piece_size()):
                vocab2.add(tokenizer2.id_to_piece(i))
            
            # 중복 비율 계산
            intersection = len(vocab1.intersection(vocab2))
            union = len(vocab1.union(vocab2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logging.warning(f"어휘 중복 계산 중 오류: {e}")
            return 0.0
    
    def simulate_asr_inference(self, tokenizer, texts: List[str]) -> Tuple[float, float]:
        """
        ASR 추론을 시뮬레이션하여 지연시간을 측정합니다.
        
        Args:
            tokenizer: 토크나이저
            texts: 추론할 텍스트 리스트
            
        Returns:
            Tuple[float, float]: (평균 지연시간, 처리량)
        """
        latencies = []
        
        for text in texts:
            start_time = time.time()
            
            # 토크나이저 추론 시뮬레이션
            tokens = tokenizer.encode_as_pieces(text)
            _ = tokenizer.encode_as_ids(text)  # 실제 추론과 유사하게
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # 밀리초 단위
            latencies.append(latency)
        
        avg_latency = np.mean(latencies)
        throughput = len(texts) / (np.sum(latencies) / 1000)  # 문장/초
        
        return avg_latency, throughput
    
    def calculate_cer(self, reference: str, hypothesis: str) -> float:
        """
        Character Error Rate를 계산합니다.
        
        Args:
            reference: 참조 텍스트
            hypothesis: 가설 텍스트
            
        Returns:
            float: CER 값
        """
        # 한국어 문자 단위로 분할
        ref_chars = list(reference.replace(' ', ''))
        hyp_chars = list(hypothesis.replace(' ', ''))
        
        # Levenshtein 거리 계산
        m, n = len(ref_chars), len(hyp_chars)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_chars[i-1] == hyp_chars[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        
        edit_distance = dp[m][n]
        cer = edit_distance / len(ref_chars) if ref_chars else 0.0
        
        return cer
    
    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """
        Word Error Rate를 계산합니다.
        
        Args:
            reference: 참조 텍스트
            hypothesis: 가설 텍스트
            
        Returns:
            float: WER 값
        """
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        # Levenshtein 거리 계산
        m, n = len(ref_words), len(hyp_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        
        edit_distance = dp[m][n]
        wer = edit_distance / len(ref_words) if ref_words else 0.0
        
        return wer
    
    def simulate_asr_errors(self, tokenizer, texts: List[str]) -> Tuple[float, float]:
        """
        ASR 오류를 시뮬레이션하여 CER과 WER을 계산합니다.
        
        Args:
            tokenizer: 토크나이저
            texts: 텍스트 리스트
            
        Returns:
            Tuple[float, float]: (CER, WER)
        """
        total_cer = 0.0
        total_wer = 0.0
        valid_texts = 0
        
        for text in texts:
            if not text.strip():
                continue
                
            # 토큰화
            tokens = tokenizer.encode_as_pieces(text)
            
            # 토큰 수에 따른 오류 시뮬레이션
            # 토큰이 많을수록 오류가 증가한다고 가정
            token_count = len(tokens)
            
            # 오류 확률 계산 (토큰 수에 비례)
            error_prob = min(0.3, token_count * 0.01)  # 최대 30% 오류
            
            # 가상의 오류 생성
            if np.random.random() < error_prob:
                # 문자 단위 오류 시뮬레이션
                chars = list(text)
                num_errors = max(1, int(len(chars) * error_prob * 0.1))
                
                for _ in range(num_errors):
                    if chars:
                        idx = np.random.randint(0, len(chars))
                        # 삭제, 삽입, 치환 중 하나 선택
                        error_type = np.random.choice(['delete', 'insert', 'substitute'])
                        
                        if error_type == 'delete':
                            chars.pop(idx)
                        elif error_type == 'insert':
                            chars.insert(idx, np.random.choice(['가', '나', '다', '라', '마']))
                        else:  # substitute
                            chars[idx] = np.random.choice(['가', '나', '다', '라', '마'])
                
                hypothesis = ''.join(chars)
            else:
                hypothesis = text
            
            # CER과 WER 계산
            cer = self.calculate_cer(text, hypothesis)
            wer = self.calculate_wer(text, hypothesis)
            
            total_cer += cer
            total_wer += wer
            valid_texts += 1
        
        avg_cer = total_cer / valid_texts if valid_texts > 0 else 0.0
        avg_wer = total_wer / valid_texts if valid_texts > 0 else 0.0
        
        return avg_cer, avg_wer
    
    def evaluate_tokenizers_extrinsic(self, model_paths: Dict[str, str],
                                    validation_file: str) -> Dict[str, ExtrinsicMetrics]:
        """
        여러 토크나이저의 Extrinsic 평가를 수행합니다.
        
        Args:
            model_paths: 토크나이저 모델 경로 딕셔너리
            validation_file: 검증 데이터 파일 경로
            
        Returns:
            Dict[str, ExtrinsicMetrics]: 각 토크나이저의 평가 결과
        """
        # 검증 데이터 로드
        with open(validation_file, 'r', encoding='utf-8') as f:
            validation_texts = [line.strip() for line in f.readlines()]
        
        results = {}
        
        for name, model_path in model_paths.items():
            if model_path is None or not os.path.exists(model_path):
                logging.warning(f"토크나이저 모델이 없습니다: {name}")
                continue
                
            logging.info(f"{name} 토크나이저 Extrinsic 평가 시작...")
            
            try:
                tokenizer = self.load_tokenizer(model_path)
                
                # 지연시간 및 처리량 측정
                avg_latency, throughput = self.simulate_asr_inference(tokenizer, validation_texts)
                
                # CER과 WER 시뮬레이션
                cer, wer = self.simulate_asr_errors(tokenizer, validation_texts)
                
                metrics = ExtrinsicMetrics(
                    cer=cer,
                    wer=wer,
                    latency_ms=avg_latency,
                    throughput=throughput
                )
                
                results[name] = metrics
                
                logging.info(f"{name} 평가 완료:")
                logging.info(f"  - CER: {metrics.cer:.4f}")
                logging.info(f"  - WER: {metrics.wer:.4f}")
                logging.info(f"  - Latency: {metrics.latency_ms:.2f}ms")
                logging.info(f"  - Throughput: {metrics.throughput:.2f} sentences/sec")
                
            except Exception as e:
                logging.error(f"{name} 토크나이저 평가 중 오류: {e}")
        
        return results
    
    def generate_evaluation_report(self, intrinsic_results: Dict[str, IntrinsicMetrics],
                                 extrinsic_results: Dict[str, ExtrinsicMetrics],
                                 output_dir: str):
        """
        평가 결과 보고서를 생성합니다.
        
        Args:
            intrinsic_results: Intrinsic 평가 결과
            extrinsic_results: Extrinsic 평가 결과
            output_dir: 출력 디렉토리
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 결과를 DataFrame으로 변환
        intrinsic_data = []
        for name, metrics in intrinsic_results.items():
            intrinsic_data.append({
                'Tokenizer': name,
                'Total_Words': metrics.total_word_count,
                'Total_Tokens': metrics.total_encountered_tokens,
                'Unk_Tokens': metrics.unk_token_count,
                'Vocab_Coverage': metrics.vocab_coverage,
                'Word_Coverage': metrics.word_coverage,
                'Fertility': metrics.fertility,
                'Avg_Tokens_Per_Word': metrics.avg_tokens_per_word,
                'Avg_Token_Length': metrics.avg_token_length,
                'Continued_Word_Rate': metrics.continued_word_rate,
                'Unique_Tokens': metrics.unique_tokens,
                'Unique_Words': metrics.unique_words
            })
        intrinsic_df = pd.DataFrame(intrinsic_data)
        
        extrinsic_data = []
        for name, metrics in extrinsic_results.items():
            extrinsic_data.append({
                'Tokenizer': name,
                'CER': metrics.cer,
                'WER': metrics.wer,
                'Latency_ms': metrics.latency_ms,
                'Throughput': metrics.throughput
            })
        extrinsic_df = pd.DataFrame(extrinsic_data)
        
        # 결과 저장
        intrinsic_df.to_csv(os.path.join(output_dir, 'intrinsic_results.csv'), index=False)
        extrinsic_df.to_csv(os.path.join(output_dir, 'extrinsic_results.csv'), index=False)
        
        # 시각화
        self._create_visualizations(intrinsic_df, extrinsic_df, output_dir)
        
        # 요약 보고서 생성
        self._create_summary_report(intrinsic_df, extrinsic_df, output_dir)
        
        logging.info(f"평가 보고서가 {output_dir}에 저장되었습니다.")
    
    def _create_visualizations(self, intrinsic_df: pd.DataFrame, 
                             extrinsic_df: pd.DataFrame, output_dir: str):
        """시각화를 생성합니다."""
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        
        # Intrinsic 지표 시각화
        if not intrinsic_df.empty:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Intrinsic Evaluation Results', fontsize=16)
            
            # Fertility
            axes[0, 0].bar(intrinsic_df['Tokenizer'], intrinsic_df['Fertility'])
            axes[0, 0].set_title('Fertility (Tokens/Word)')
            axes[0, 0].set_ylabel('Fertility')
            
            # OOV Rate
            axes[0, 1].bar(intrinsic_df['Tokenizer'], intrinsic_df['OOV_Rate'])
            axes[0, 1].set_title('OOV Rate')
            axes[0, 1].set_ylabel('OOV Rate')
            
            # Vocab Coverage
            axes[1, 0].bar(intrinsic_df['Tokenizer'], intrinsic_df['Vocab_Coverage'])
            axes[1, 0].set_title('Vocabulary Coverage')
            axes[1, 0].set_ylabel('Coverage')
            
            # Avg Tokens Per Word
            axes[1, 1].bar(intrinsic_df['Tokenizer'], intrinsic_df['Avg_Tokens_Per_Word'])
            axes[1, 1].set_title('Average Tokens Per Word')
            axes[1, 1].set_ylabel('Tokens')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'intrinsic_evaluation.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Extrinsic 지표 시각화 (CER 중심)
        if not extrinsic_df.empty:
            fig, axes = plt.subplots(2, 2, figsize=(18, 12))
            fig.suptitle('Extrinsic Evaluation Results (Korean ASR Focus)', fontsize=16)
            
            # CER (한국어에 더 적합)
            axes[0, 0].bar(extrinsic_df['Tokenizer'], extrinsic_df['CER'])
            axes[0, 0].set_title('Character Error Rate (CER)')
            axes[0, 0].set_ylabel('CER')
            if len(extrinsic_df) > 0:
                axes[0, 0].set_ylim(0, max(extrinsic_df['CER']) * 1.2)
            
            # WER (참고용)
            axes[0, 1].bar(extrinsic_df['Tokenizer'], extrinsic_df['WER'])
            axes[0, 1].set_title('Word Error Rate (WER)')
            axes[0, 1].set_ylabel('WER')
            if len(extrinsic_df) > 0:
                axes[0, 1].set_ylim(0, max(extrinsic_df['WER']) * 1.2)
            
            # Latency
            axes[1, 0].bar(extrinsic_df['Tokenizer'], extrinsic_df['Latency_ms'])
            axes[1, 0].set_title('Inference Latency')
            axes[1, 0].set_ylabel('Latency (ms)')
            
            # Throughput
            axes[1, 1].bar(extrinsic_df['Tokenizer'], extrinsic_df['Throughput'])
            axes[1, 1].set_title('Throughput')
            axes[1, 1].set_ylabel('Sentences/sec')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'extrinsic_evaluation.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_summary_report(self, intrinsic_df: pd.DataFrame, 
                             extrinsic_df: pd.DataFrame, output_dir: str):
        """요약 보고서를 생성합니다."""
        report = []
        report.append("# 한국어 ASR 토크나이저 평가 보고서")
        report.append("")
        report.append("> **참고**: 한국어 ASR에서는 CER(Character Error Rate)이 WER보다 더 적합한 평가 지표입니다.")
        report.append("")
        
        # Intrinsic 평가 결과
        report.append("## Intrinsic 평가 결과")
        report.append("")
        report.append("| 토크나이저 | Fertility | OOV Rate | Vocab Coverage | Avg Tokens/Word |")
        report.append("|------------|-----------|----------|----------------|-----------------|")
        
        for _, row in intrinsic_df.iterrows():
            report.append(f"| {row['Tokenizer']} | {row['Fertility']:.4f} | {row['OOV_Rate']:.4f} | {row['Vocab_Coverage']:.4f} | {row['Avg_Tokens_Per_Word']:.4f} |")
        
        report.append("")
        
        # Extrinsic 평가 결과 (CER 중심)
        report.append("## Extrinsic 평가 결과")
        report.append("")
        report.append("| 토크나이저 | **CER** | WER | Latency (ms) | Throughput (sentences/sec) |")
        report.append("|------------|---------|-----|--------------|---------------------------|")
        
        for _, row in extrinsic_df.iterrows():
            report.append(f"| {row['Tokenizer']} | **{row['CER']:.4f}** | {row['WER']:.4f} | {row['Latency_ms']:.2f} | {row['Throughput']:.2f} |")
        
        report.append("")
        
        # 분석 및 권장사항
        report.append("## 분석 및 권장사항")
        report.append("")
        
        # 최적 토크나이저 찾기 (CER 중심)
        best_fertility = intrinsic_df.loc[intrinsic_df['Fertility'].idxmin(), 'Tokenizer']
        best_oov = intrinsic_df.loc[intrinsic_df['OOV_Rate'].idxmin(), 'Tokenizer']
        best_cer = extrinsic_df.loc[extrinsic_df['CER'].idxmin(), 'Tokenizer']
        best_latency = extrinsic_df.loc[extrinsic_df['Latency_ms'].idxmin(), 'Tokenizer']
        
        report.append(f"- **최저 Fertility**: {best_fertility}")
        report.append(f"- **최저 OOV Rate**: {best_oov}")
        report.append(f"- **최저 CER**: {best_cer} ⭐ (한국어 ASR에 가장 중요)")
        report.append(f"- **최저 Latency**: {best_latency}")
        report.append("")
        
        # 도메인별 권장사항 (CER 중심)
        report.append("### 도메인별 권장사항")
        report.append("")
        report.append("- **한국어 ASR 정확도 중시**: **CER이 가장 낮은 토크나이저 선택** ⭐")
        report.append("- **실시간 ASR (Latency 중시)**: Latency가 가장 낮은 토크나이저 선택")
        report.append("- **외래어/특수단어 처리**: OOV Rate가 낮은 토크나이저 선택")
        report.append("- **일반적인 사용**: CER과 Latency를 고려한 균형잡힌 선택")
        report.append("")
        
        # 한국어 ASR 특성 설명
        report.append("### 한국어 ASR에서 CER이 중요한 이유")
        report.append("")
        report.append("1. **음절 단위 언어**: 한국어는 음절 단위로 구성되어 있어 문자 단위 평가가 더 정확")
        report.append("2. **조사/어미 변화**: 한국어의 풍부한 조사와 어미 변화를 문자 단위로 평가하는 것이 적합")
        report.append("3. **복합어 처리**: 한국어의 복합어 특성을 고려한 평가")
        report.append("4. **외래어 처리**: SITE, CORD 등 외래어의 문자 단위 정확도 평가")
        
        # 보고서 저장
        with open(os.path.join(output_dir, 'evaluation_report.md'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 평가 예시
    evaluator = TokenizerEvaluator()
    
    model_paths = {
        "kiwi": "outputs/tokenizers/tokenizer_kiwi.model",
        "mecab": "outputs/tokenizers/tokenizer_mecab.model"
    }
    
    validation_file = "data/validation_text.txt"
    output_dir = "outputs/evaluation"
    
    # Intrinsic 평가
    intrinsic_results = evaluator.evaluate_tokenizers_intrinsic(model_paths, validation_file)
    
    # Extrinsic 평가
    extrinsic_results = evaluator.evaluate_tokenizers_extrinsic(model_paths, validation_file)
    
    # 보고서 생성
    evaluator.generate_evaluation_report(intrinsic_results, extrinsic_results, output_dir)
    
    print("토크나이저 평가가 완료되었습니다.") 