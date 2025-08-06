import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np

import hydra
from rouge import Rouge # 모델의 성능을 평가하기 위한 라이브러리입니다.
from peft import get_peft_model, LoraConfig, TaskType


class BartSummarizationModule(pl.LightningModule):
    def __init__(self, cfg, model, tokenizer):
        super().__init__()
        self.cfg = cfg

        """ # PEFT (LoRA) 설정 정의
        # r: LoRA의 rank. 낮을수록 파라미터 수가 적어지며, 보통 8, 16, 32 등을 사용.
        # lora_alpha: LoRA 스케일링 인자. 보통 r의 2배 또는 4배 값을 사용.
        # target_modules: LoRA를 적용할 레이어 지정 (KoBART에서는 q_proj, v_proj).
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=32,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"]
        )

        # get_peft_model 함수로 모델을 감싸 PEFT 모델로 변환
        self.model = get_peft_model(model, peft_config)
        print("*"*30)
        print(self.model)
        # 학습 가능한 파라미터 수와 비율 확인 (PEFT의 효과를 직접 확인)
        self.model.print_trainable_parameters()
        # 출력 예시: trainable params: 368,640 || all params: 124,334,208 || trainable%: 0.296...
        print("*"*30) """

        self.model = model
        self.tokenizer = tokenizer
        self.save_hyperparameters(ignore=['model', 'tokenizer'])

        # v2.0을 위한 출력 저장소
        self.validation_outputs = []
        self.test_outputs = []
        
    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Generation for metrics (BLEU, ROUGE)
        if self.cfg.trainer.predict_with_generate:
            generated_ids = self.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                #max_length=self.cfg.tokenizer.decoder_max_len,
                min_new_tokens=self.cfg.tokenizer.decoder_min_len,
                max_new_tokens=self.cfg.tokenizer.decoder_max_len,
                num_beams=5,
                early_stopping=True,
                
                # 매우 보수적 샘플링
                # do_sample=False,
                # temperature=0.6,      # 낮은 창의성, 높은 정확성
                # top_p=0.8,           # 상위 80%만 고려
                # top_k=30,            # 제한적 후보
                
                # 반복 최소화
                #repetition_penalty=1.1,
                #no_repeat_ngram_size=3,
                
                # 길이 페널티 (요약 품질 향상)
                #length_penalty=1.2,   # 적절한 길이 유도
                
                # 토큰 설정
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            self.validation_outputs.append({
                'generated_ids': generated_ids,
                'labels': batch['labels']
            })
        
        return {'val_loss': loss}
    
    def test_step(self, batch, batch_idx):
        generated_ids = self.model.generate(input_ids=batch['input_ids'],
                            #max_length=self.cfg.tokenizer.decoder_max_len,                    
                            min_new_tokens=self.cfg.tokenizer.decoder_min_len,
                            max_new_tokens=self.cfg.tokenizer.decoder_max_len,
                            num_beams=5,
                            early_stopping=True,
                                                        
                            # 매우 보수적 샘플링
                            # do_sample=False,
                            # temperature=0.6,      # 낮은 창의성, 높은 정확성
                            # top_p=0.8,           # 상위 80%만 고려
                            # top_k=30,            # 제한적 후보
                            
                            # 반복 최소화
                            #repetition_penalty=1.1,
                            #no_repeat_ngram_size=3,
                            
                            # 길이 페널티 (요약 품질 향상)
                            #length_penalty=1.0,   # 적절한 길이 유도
                            
                            # 토큰 설정
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
         
        for ids in generated_ids:
            result = self.tokenizer.decode(ids)
            self.test_outputs.append(result)

        return result    

    def on_validation_epoch_end(self):

        if not self.validation_outputs:
            return
        
        # 패딩을 통한 크기 통일
        all_generated_list = [x['generated_ids'] for x in self.validation_outputs]
        all_labels_list = [x['labels'] for x in self.validation_outputs]
        
        # 최대 길이 계산
        max_gen_len = max(gen.size(-1) for gen in all_generated_list)
        max_label_len = max(label.size(-1) for label in all_labels_list)
        
        # 패딩 처리
        padded_generated = []
        padded_labels = []
        
        for gen, label in zip(all_generated_list, all_labels_list):
            gen_padded = F.pad(gen, (0, max_gen_len - gen.size(-1)), value=self.tokenizer.pad_token_id)
            label_padded = F.pad(label, (0, max_label_len - label.size(-1)), value=self.tokenizer.pad_token_id)
            padded_generated.append(gen_padded)
            padded_labels.append(label_padded)
        
        all_generated = torch.cat(padded_generated, dim=0)
        all_labels = torch.cat(padded_labels, dim=0)
        
        # Compute metrics (you'll need to implement compute_metrics function)
        metrics = self.compute_metrics(all_generated, all_labels)
        for key, value in metrics.items():
            self.log(f'val_{key}', value, prog_bar=True)

        # 다음 에포크를 위해 초기화
        self.validation_outputs.clear()    

    def on_test_epoch_end(self):

        # 정확한 평가를 위하여 노이즈에 해당되는 스페셜 토큰을 제거합니다.
        remove_tokens = [str(token) for token in self.cfg.tokenizer.remove_tokens]
        preprocessed_summary = self.test_outputs.copy()
        for token in remove_tokens:
            preprocessed_summary = [sentence.replace(token," ") for sentence in preprocessed_summary]

        self.test_outputs = preprocessed_summary
        
    # 모델 성능에 대한 평가 지표를 정의합니다. 본 대회에서는 ROUGE 점수를 통해 모델의 성능을 평가합니다.
    def compute_metrics(self, all_generated, all_labels):
        rouge = Rouge()
        predictions = all_generated
        labels = all_labels

        predictions[predictions == -100] = self.tokenizer.pad_token_id
        labels[labels == -100] = self.tokenizer.pad_token_id

        decoded_preds = self.tokenizer.batch_decode(predictions, clean_up_tokenization_spaces=True)
        labels = self.tokenizer.batch_decode(labels, clean_up_tokenization_spaces=True)

        # 정확한 평가를 위해 미리 정의된 불필요한 생성토큰들을 제거합니다.
        replaced_predictions = decoded_preds.copy()
        replaced_labels = labels.copy()
        remove_tokens = [str(token) for token in self.cfg.tokenizer.remove_tokens]
        for token in remove_tokens:
            replaced_predictions = [sentence.replace(token," ") for sentence in replaced_predictions]
            replaced_labels = [sentence.replace(token," ") for sentence in replaced_labels]

        for i in range(0, 10):
            print('-'*150)
            print(f"PRED: {replaced_predictions[i]}")
            print(f"GOLD: {replaced_labels[i]}")
            print(f"similarity="+str(self.calculate_semantic_similarity(replaced_labels[i], replaced_predictions[i])))


        # 최종적인 ROUGE 점수를 계산합니다.
        results = rouge.get_scores(replaced_predictions, replaced_labels,avg=True)

        # ROUGE 점수 중 F-1 score를 통해 평가합니다.
        result = {key: value["f"] for key, value in results.items()}
        return result
    
    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.cfg.optimizer, params=self.parameters() )       

        # 스케줄러 생성
        scheduler = hydra.utils.instantiate(self.cfg.scheduler, optimizer=optimizer)
        
        return {
            'optimizer': optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",     # 에포크 단위 업데이트
                "frequency": 1,          # 매 에포크마다
                "monitor": None,         # 자동 스케줄링이므로 불필요
                "strict": False,         # 모니터링 없으므로 False
                "name": "cosine_annealing_lr"
            }
        }

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
