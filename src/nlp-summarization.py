""" Dialogue Summarization** 경진대회에 오신 여러분 환영합니다!   
본 대회에서는 최소 2명에서 최대 7명이 등장하여 나누는 대화를 요약하는 BART 기반 모델의 baseline code를 제공합니다.     
주어진 데이터를 활용하여 일상 대화에 대한 요약을 효과적으로 생성하는 모델을 만들어봅시다!
 """

import os
import sys
import datetime

import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset , DataLoader

from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig
""" from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback """

import wandb # 모델 학습 과정을 손쉽게 Tracking하고, 시각화할 수 있는 라이브러리입니다.
from rouge import Rouge # 모델의 성능을 평가하기 위한 라이브러리입니다.
from dotenv import load_dotenv, dotenv_values

import hydra
from omegaconf import DictConfig

# 하이드라와 주피터 노트북은 아규먼트 관련 충돌이 발생하므로 초기화 해줌
sys.argv = ['']
sys.path.append('.')  

from src.data.bart_summarization_dataset import get_datasets
from src import config
from src.models.bart_summarization_module import BartSummarizationModule

def generate_experiment_name(model_name, learning_rate, batch_size, additional_info=""):
    """동적 실험 이름 생성"""
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    
    base_name = f"{model_name.replace('/', '_')}_lr{learning_rate}_bs{batch_size}"
    
    if additional_info:
        base_name += f"_{additional_info}"
    
    return f"{base_name}_{timestamp}"    


# 데이터 준비 함수
def prepare_data(cfg, tokenizer):
    
   # 데이터셋 생성
    train_dataset, val_dataset, test_dataset = get_datasets(cfg, tokenizer)

    """ print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))
    sample_0 = train_dataset[0]
    print(sample_0) """


    # DataLoader 정의
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,  # 별도의 검증 데이터셋
        batch_size=cfg.data.batch_size,
        shuffle=False,  # 검증 시에는 셔플하지 않음
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader

# 학습을 위한 tokenizer와 사전 학습된 모델을 불러옵니다.
def load_tokenizer_and_model_for_train(cfg):
    print('-'*10, 'Load tokenizer & model', '-'*10,)
    print('-'*10, f'Model Name : {cfg.model.model_name}', '-'*10,)
    model_name = cfg.model.model_name
    
    bart_config = BartConfig().from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generate_model = BartForConditionalGeneration.from_pretrained(model_name,config=bart_config)

    special_tokens_dict={'additional_special_tokens':[str(token) for token in cfg.tokenizer.special_tokens]}
    tokenizer.add_special_tokens(special_tokens_dict)

    generate_model.resize_token_embeddings(len(tokenizer)) # 사전에 special token을 추가했으므로 재구성 해줍니다.
    print(generate_model.config)

    print('-'*10, 'Load tokenizer & model complete', '-'*10,)
    return generate_model , tokenizer


def test(trainer, model, test_loader):

    # 테스트
    trainer.test(model, test_loader)

    print("테스트 갯수=",len(model.test_outputs))
    
    if len(model.test_outputs) > 0:
        # 모든 예측값과 실제값 합치기
        all_preds = model.test_outputs
        pred_df = test_loader.dataset.df

        output = pd.DataFrame(
            {
                "fname": pred_df['fname'],
                "summary" : all_preds,
            }
        )

        output.to_csv("output.csv", index=False)
        
    else:
        print("테스트 결과를 가져올 수 없습니다.")


@hydra.main(config_path=str(config.CONFIGS_DIR), config_name="config", version_base=None)
def main(cfg):

     # cfg 에서 최상위 model 키값을 뺀다.
    cfg = cfg.model

    # WandB Logger 초기화
    wandb_logger = WandbLogger(
        project= str(dotenv_values().get('WANDB_PROJECT')),                                                                                        # 프로젝트 이름
        name=generate_experiment_name(cfg.model.model_name, cfg.optimizer.lr, cfg.data.batch_size),                   # 실험 이름 (선택사항)
        job_type="train",                                                                                                   # 작업 타입 (선택사항)
        save_dir=str(config.LOG_ROOT_DIR),
        log_model=True

    )

    pl.seed_everything(cfg.custom.seed_num)

    # 사용할 모델과 tokenizer를 불러옵니다.
    generate_model , tokenizer = load_tokenizer_and_model_for_train(cfg)
    print('-'*10,"tokenizer special tokens : ",tokenizer.special_tokens_map,'-'*10)

    # 데이터 로더 준비
    train_loader, val_loader, test_loader = prepare_data(cfg, tokenizer)


    # Lightning Module 생성
    lightning_module = BartSummarizationModule(cfg, generate_model, tokenizer)

    # 콜백을 직접 생성
    early_stopping = hydra.utils.instantiate(cfg.callbacks.early_stopping)
    lr_monitor = hydra.utils.instantiate(cfg.callbacks.lr_monitor)
    model_checkpoint = hydra.utils.instantiate(cfg.callbacks.model_checkpoint)

    callbacks = [early_stopping, lr_monitor, model_checkpoint]

    trainer = Trainer(
        default_root_dir=str(config.CHECKPOINT_DIR), 
        max_epochs=cfg.trainer.max_epochs, 
        accelerator=cfg.trainer.accelerator, 
        callbacks=callbacks, 
        logger=wandb_logger, 
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps)
    

    # 훈련
    if cfg.custom.do_checkpoint == True and os.path.exists(cfg.custom.ckpt_path):
        trainer.fit(lightning_module, train_loader, val_loader, ckpt_path=cfg.custom.ckpt_path)
    else:
        trainer.fit(lightning_module, train_loader, val_loader)
    
    # 테스트
    if(cfg.custom.do_test == True): 
        test(trainer, lightning_module, test_loader)


    wandb.finish()


# poetry run python src/nlp-summarization.py
if __name__ == "__main__":

    main()