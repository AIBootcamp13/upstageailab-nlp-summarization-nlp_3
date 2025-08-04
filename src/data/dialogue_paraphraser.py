import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import re

import numpy as np

class DialogueParaphraser:
    """
    화자 정보를 완벽하게 보존하면서 대화를 패러프레이징하는 클래스
    """
    def __init__(self):
        self.paraphraser = None
        self.model_name = "psyche/KoT5-paraphrase-generation" # 사용하시려는 모델
        
        try:
            print(f"'{self.model_name}' 모델 로딩 시도 중...")
            self.paraphraser = pipeline(
                "text2text-generation",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            print(f"✅ '{self.model_name}' 모델 로딩 성공!")

        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")

    def paraphrase_dialogue(self, dialogue: str) -> str:
        """
        대화 구조를 보존하며 안전하게 패러프레이징합니다.
        """
        # 1. 정규표현식을 사용해 화자 태그와 발화 내용을 정확히 분리합니다.
        speaker_pattern = r'(#(?:Person|개인)\d+#:)'
        parts = re.split(speaker_pattern, dialogue)
        
        segments = []
        # parts 리스트는 [ '', '#Person1#:', ' 발화 내용1 ', '#Person2#:', ' 발화 내용2' ] 형태로 나옵니다.
        for i in range(1, len(parts), 2):
            speaker_tag = parts[i]
            utterance = parts[i+1].strip()
            if utterance:
                segments.append((speaker_tag, utterance))

        paraphrased_segments = []
        for speaker_tag, utterance in segments:
            try:
                # 2. 순수한 발화 내용(utterance)만 모델에 전달합니다.
                #    모델이 'summarization'일 경우를 대비해 태스크 이름을 확인합니다.
                
                # text2text-generation 모델 사용
                result = self.paraphraser(
                    utterance,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.8,          # 더 높은 온도로 창의성 증가
                    top_k=50,                 # top_k 값을 설정하여 다양성 확보
                    repetition_penalty=1.2,   # 반복을 줄여 새로운 표현 유도
                    num_beams=1,              # 빔 서치 대신 샘플링에 집중
                    num_return_sequences=1
                )
                paraphrased_utterance = result[0]['generated_text']

                # 3. 패러프레이징된 결과에 원래의 화자 태그를 다시 붙여줍니다.
                paraphrased_segments.append(f"{speaker_tag} {paraphrased_utterance}")

            except Exception:
                # 패러프레이징 실패 시, 안전하게 원본을 사용합니다.
                paraphrased_segments.append(f"{speaker_tag} {utterance}")
        
        # 4. 개행문자(\n)로 각 발화를 연결하여 최종 대화를 만듭니다.
        return "\n".join(paraphrased_segments)
    
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """의미 기반 유사도 계산 (기존 Jaccard 대신)"""
        
        # 모델이 없다면 초기화
        if not hasattr(self, 'similarity_model'):
            print("유사도 측정 모델 로딩 (최초 1회)...")
            self.similarity_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        
        # 문장을 임베딩 벡터로 변환
        embeddings = self.similarity_model.encode([text1, text2])
        
        # 코사인 유사도 계산
        vec1 = embeddings[0]
        vec2 = embeddings[1]
        
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        return similarity
   

    def paraphrase_dialogue_batch(self, dialogues_list):
        """배치 단위로 패러프레이징 처리 (화자 처리 로직 포함)"""
        try:
            # 1. 각 대화를 화자별 발화로 분리
            all_utterances = []  # 모든 순수 발화 내용
            dialogue_structures = []  # 각 대화의 구조 정보
            
            speaker_pattern = r'(#(?:Person|개인)\d+#:)'
            
            for dialogue_idx, dialogue in enumerate(dialogues_list):
                parts = re.split(speaker_pattern, dialogue)
                
                segments = []
                utterances_in_dialogue = []
                
                # 화자 태그와 발화 내용 분리
                for i in range(1, len(parts), 2):
                    if i+1 < len(parts):
                        speaker_tag = parts[i]
                        utterance = parts[i+1].strip()
                        if utterance:
                            segments.append((speaker_tag, utterance))
                            utterances_in_dialogue.append(utterance)
                            all_utterances.append(utterance)
                
                dialogue_structures.append({
                    'dialogue_idx': dialogue_idx,
                    'segments': segments,
                    'utterance_count': len(utterances_in_dialogue)
                })
            
            # 2. 모든 순수 발화 내용을 배치로 패러프레이징
            if all_utterances:
                paraphrased_utterances = self.paraphraser(
                    all_utterances,
                    batch_size=min(len(all_utterances), 16),  # 메모리 고려하여 배치 크기 제한
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.8,
                    top_k=50,
                    repetition_penalty=1.2,
                    num_beams=1,
                    num_return_sequences=1
                )
                
                # 결과 형식 통일
                if isinstance(paraphrased_utterances[0], dict):
                    paraphrased_utterances = [result.get('generated_text', result.get('text', '')) 
                                            for result in paraphrased_utterances]
                else:
                    paraphrased_utterances = [str(result) for result in paraphrased_utterances]
            else:
                paraphrased_utterances = []
            
            # 3. 패러프레이징된 발화를 원래 대화 구조로 재조립
            results = []
            utterance_idx = 0
            
            for structure in dialogue_structures:
                paraphrased_segments = []
                
                for speaker_tag, original_utterance in structure['segments']:
                    if utterance_idx < len(paraphrased_utterances):
                        paraphrased_utterance = paraphrased_utterances[utterance_idx]
                        paraphrased_segments.append(f"{speaker_tag} {paraphrased_utterance}")
                        utterance_idx += 1
                    else:
                        # 패러프레이징 실패 시 원본 사용
                        paraphrased_segments.append(f"{speaker_tag} {original_utterance}")
                
                # 각 대화를 개행문자로 연결
                reconstructed_dialogue = "\n".join(paraphrased_segments)
                results.append(reconstructed_dialogue)
            
            return results
            
        except Exception as e:
            print(f"배치 패러프레이징 오류: {e}")
            import traceback
            traceback.print_exc()
            
            # 실패시 개별 처리로 fallback
            return [self.paraphrase_dialogue(dialogue) for dialogue in dialogues_list]
    
    
    def calculate_semantic_similarity_batch(self, original_list, paraphrased_list):
        """배치 단위로 유사도 계산"""
        similarities = []
        for orig, para in zip(original_list, paraphrased_list):
            try:
                similarity = self.calculate_semantic_similarity(orig, para)
                similarities.append(similarity)
            except Exception as e:
                print(f"유사도 계산 오류: {e}")
                similarities.append(0.0)  # 오류시 낮은 점수
        return similarities    