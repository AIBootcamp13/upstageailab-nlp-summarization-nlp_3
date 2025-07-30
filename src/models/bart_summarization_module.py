import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd

import hydra
from rouge import Rouge # 모델의 성능을 평가하기 위한 라이브러리입니다.

class BartSummarizationModule(pl.LightningModule):
    def __init__(self, cfg, model, tokenizer):
        super().__init__()
        self.cfg = cfg
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
                max_length=self.cfg.trainer.generation_max_length,
                num_beams=4,
                early_stopping=True
            )
            
            self.validation_outputs.append({
                'generated_ids': generated_ids,
                'labels': batch['labels']
            })
        
        return {'val_loss': loss}
    
    def test_step(self, batch, batch_idx):
        generated_ids = self.model.generate(input_ids=batch['input_ids'],
                            no_repeat_ngram_size=3,
                            early_stopping=True,
                            max_length=self.cfg.trainer.generation_max_length,
                            num_beams=4
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

        print('-'*150)
        print(f"PRED: {replaced_predictions[0]}")
        print(f"GOLD: {replaced_labels[0]}")
        print('-'*150)
        print(f"PRED: {replaced_predictions[1]}")
        print(f"GOLD: {replaced_labels[1]}")
        print('-'*150)
        print(f"PRED: {replaced_predictions[2]}")
        print(f"GOLD: {replaced_labels[2]}")

        # 최종적인 ROUGE 점수를 계산합니다.
        results = rouge.get_scores(replaced_predictions, replaced_labels,avg=True)

        # ROUGE 점수 중 F-1 score를 통해 평가합니다.
        result = {key: value["f"] for key, value in results.items()}
        return result
    
    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.cfg.optimizer, params=self.parameters() )       
        return [optimizer]


