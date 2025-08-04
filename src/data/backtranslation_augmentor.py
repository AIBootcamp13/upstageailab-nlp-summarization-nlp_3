import re

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
from typing import List

class BackTranslationAugmentor:
    def __init__(self, use_gpu: bool = True):
        print("NLLB-200-distilled-600M 모델 로딩 중...")
        
        # GPU 사용 설정
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"사용 디바이스: {self.device}")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        
        # 모델 로드 및 GPU로 이동
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        self.model.to(self.device)  # GPU로 모델 이동
        
        # GPU 메모리 최적화 (선택사항)
        if self.device.type == "cuda":
            self.model.half()  # FP16 사용으로 메모리 절약
            print(f"GPU 메모리 사용량: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        # 언어 코드
        self.ko_lang_code = "kor_Hang"
        self.en_lang_code = "eng_Latn"

        self.ko_token_id = self._get_lang_token_id(self.ko_lang_code)
        self.en_token_id = self._get_lang_token_id(self.en_lang_code)
        print("모델 로딩 완료!")
    
    def _get_lang_token_id(self, lang_code: str) -> int:
        """NLLB 언어 토큰 ID 안전하게 가져오기"""
        
        # 방법 1: 토크나이저 어휘집에서 직접 찾기
        if hasattr(self.tokenizer, 'get_vocab'):
            vocab = self.tokenizer.get_vocab()
            if lang_code in vocab:
                return vocab[lang_code]
        
        # 방법 2: 특수 토큰으로 인코딩
        try:
            encoded = self.tokenizer.encode(
                lang_code, 
                add_special_tokens=False, 
                return_tensors="pt"
            )
            if len(encoded[0]) > 0:
                return encoded[0][0].item()
        except:
            pass
        
        # 방법 3: 더미 텍스트와 함께 인코딩하여 BOS 토큰 찾기
        try:
            self.tokenizer.src_lang = lang_code
            encoded = self.tokenizer.encode("test", return_tensors="pt")
            return encoded[0][0].item()
        except:
            pass
        
        raise ValueError(f"언어 코드 '{lang_code}'의 토큰 ID를 찾을 수 없습니다.")

    def back_translate(self, korean_text: str, temperature: float = 0.8) -> str:
        """GPU 최적화된 백번역"""
        #print("korean_text:"+korean_text)
        try:
            # 1단계: 한국어 → 영어
            ko_inputs = self.tokenizer(
                korean_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # 입력 텐서를 GPU로 이동
            ko_inputs = {k: v.to(self.device) for k, v in ko_inputs.items()}
            
            with torch.no_grad():
                en_outputs = self.model.generate(
                    **ko_inputs,
                    forced_bos_token_id=self.en_token_id,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    num_beams=4,
                    length_penalty=1.0,  # 길이 페널티 추가
                )
            
            english_text = self.tokenizer.decode(en_outputs[0], skip_special_tokens=True)
            #print("english_text:"+english_text)
            
            # 2단계: 영어 → 한국어
            en_inputs = self.tokenizer(
                english_text,
                return_tensors="pt", 
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # 입력 텐서를 GPU로 이동
            en_inputs = {k: v.to(self.device) for k, v in en_inputs.items()}
            
            with torch.no_grad():
                ko_outputs = self.model.generate(
                    **en_inputs,
                    forced_bos_token_id=self.ko_token_id,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    num_beams=4,
                    length_penalty=1.0,  # 길이 페널티 추가
                )
            
            back_translated_text = self.tokenizer.decode(ko_outputs[0], skip_special_tokens=True)
            
            #print("kor_text:"+back_translated_text)
            return back_translated_text
            
        except Exception as e:
            print(f"번역 오류: {e}")
            return korean_text
        
    # skip_special_tokens 설정과 무관하게 화자 보존 필요
    def preserve_speaker_in_segment(self, korean_text: str) -> str:
        """화자 태그 보존 번역"""
        
        # 방법 1: 화자 부분과 내용 부분 분리 번역
        import re
        match = re.match(r'(#Person\d+#:)\s*(.*)', korean_text)
        
        if match:
            speaker_tag = match.group(1)  # #Person1#:
            content = match.group(2)      # 실제 내용
            
            # 내용만 번역
            translated_content = self.back_translate(content)
            
            # 화자 태그 재결합
            return f"{speaker_tag} {translated_content}"
        
        return self.back_translate(korean_text)    
    
    def split_dialogue_by_speakers(self, dialogue_text: str) -> List[str]:  
        """화자별로 대화 분할"""
        import re
    
    
        # 화자 패턴 찾기 (#Person1#, #Person2#, #개인1# 등)
        speaker_pattern = r'(#(?:Person|개인)\d+#:)'
        
        # 화자 기준으로 분할
        parts = re.split(speaker_pattern, dialogue_text)
        
        # 화자 + 발화 쌍으로 재구성
        dialogue_segments = []
        
        for i in range(1, len(parts), 2):  # 홀수 인덱스는 화자, 짝수는 발화
            if i + 1 < len(parts):
                speaker = parts[i].strip()
                utterance = parts[i + 1].strip()
                
                if utterance:  # 빈 발화가 아닌 경우만
                    dialogue_segments.append(f"{speaker} {utterance}")
        
        return dialogue_segments
    

    
    def _split_translate(self, korean_text: str, temperature: float) -> str:
        """실제 분할 번역 로직"""
        
        # 1단계: 화자별로 분할
        segments = self.split_dialogue_by_speakers(korean_text)
        print(f"화자별 분할 결과: {len(segments)}개 세그먼트")
        
        for i, seg in enumerate(segments):
            print(f"  세그먼트 {i+1}: {seg[:50]}...")
        
        translated_segments = []
            
        for segment in segments:
           
            # 일반적인 경우: 개별 번역
            #translated_segments.append(self.back_translate(segment, temperature))

            # 화자와 내용 분리 → 내용만 번역 → 화자 재부착
            preserved_segment = self.preserve_speaker_in_segment(segment)
            translated_segments.append(preserved_segment)    


        
        return ' '.join(translated_segments)

    def back_translate_with_splitting(self, korean_text: str, temperature: float = 0.8) -> str:
        """분할 백번역 메인 함수"""
        
        try:
            return self._split_translate(korean_text, temperature)
                
        except Exception as e:
            print(f"분할 번역 오류: {e}")
            return korean_text

    def augment_dialogue_data(self, original_data: list, augment_ratio: float = 1.0) -> list:
        """GPU 메모리 관리가 개선된 데이터 증강"""
        augmented_data = []
        total_samples = len(original_data)
        augment_count = int(total_samples * augment_ratio)
        
        print(f"총 {total_samples}개 중 {augment_count}개 샘플을 백번역으로 증강합니다...")
        print(f"사용 디바이스: {self.device}")
        
        # 랜덤 샘플링
        sample_indices = np.random.choice(
            total_samples,
            size=augment_count,
            replace=False
        )
        
        successful_augmentations = 0
        
        for i, idx in enumerate(sample_indices):
            if i % 50 == 0:
                print(f"진행률: {i}/{augment_count} ({i/augment_count*100:.1f}%)")
                
                # GPU 메모리 상태 출력
                if self.device.type == "cuda":
                    allocated = torch.cuda.memory_allocated() / 1e9
                    cached = torch.cuda.memory_reserved() / 1e9
                    print(f"GPU 메모리: {allocated:.1f}GB 사용 중, {cached:.1f}GB 예약됨")
            
            original_item = original_data[idx]

            try:
                # 백번역 수행
                augmented_dialogue = self.back_translate_with_splitting(
                    original_item['dialogue'],
                    temperature=0.7
                )
                
                # 품질 검증
                similarity = self.calculate_similarity(
                    original_item['dialogue'],
                    augmented_dialogue
                )
                
                # 적절한 유사도 범위 확인
                if 0.5 <= similarity <= 0.9:
                    augmented_data.append({
                        'dialogue': augmented_dialogue,
                        'summary': original_item['summary'],
                        'fname': f'train_aug_{idx}',
                        'topic': original_item['topic'],
                        'similarity': similarity
                    })
                    
                    pattern = r'(#(?:Person|개인)\d+#:)'
                    replacement = '\n\\1'

                    formatted_dialogue = re.sub(pattern, replacement, augmented_dialogue)


                    print('-'*150)
                    print(f"분할 백번역 성공:")
                    print(f"AUG: {formatted_dialogue}")
                    print(f"ORI: {original_item['dialogue']}")
                    print(f"유사도: {similarity:.3f}")
                    print('-'*150)

                    successful_augmentations += 1
                    
                # GPU 메모리 정리
                if self.device.type == "cuda" and i % 10 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"샘플 {idx} 분할 번역 실패: {e}")
                continue
        
        print(f"백번역 완료: {successful_augmentations}/{augment_count} 성공")
        
        return original_data + augmented_data
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Jaccard 유사도 계산"""
        tokens1 = set(text1.split())
        tokens2 = set(text2.split())
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        if len(union) == 0:
            return 0.0
        
        return len(intersection) / len(union)
    
    def __del__(self):
        """소멸자에서 GPU 메모리 정리"""
        if hasattr(self, 'device') and self.device.type == "cuda":
            torch.cuda.empty_cache()
