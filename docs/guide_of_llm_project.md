## 🏗️ 프로젝트 구조와 환경 설정

LLM 모델링 프로젝트를 체계적으로 관리하기 위한 디렉토리 구조를 먼저 살펴보자.

```
llm-modeling/
├── pyproject.toml              # Poetry 설정
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── models/                 # Lightning 모듈들
│   │   ├── __init__.py
│   │   ├── base_llm.py        # 기본 LLM 모듈
│   │   ├── classification.py   # 분류 태스크
│   │   ├── generation.py      # 텍스트 생성
│   │   └── qa.py              # 질문 답변
│   ├── data/                  # 데이터 모듈들
│   │   ├── __init__.py
│   │   ├── base_datamodule.py
│   │   └── text_datasets.py
│   ├── callbacks/             # 커스텀 콜백들
│   │   ├── __init__.py
│   │   ├── model_monitoring.py
│   │   └── text_generation.py
│   ├── utils/                 # 유틸리티 함수들
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── tokenization.py
│   └── experiments/           # 실험 스크립트들
│       ├── __init__.py
│       ├── train_classifier.py
│       └── train_generator.py
├── configs/                   # 설정 파일들
│   ├── model/
│   ├── data/
│   └── training/
├── notebooks/                 # Jupyter 노트북들
├── tests/                     # 테스트 파일들
└── outputs/                   # 실험 결과물들
    ├── checkpoints/
    ├── logs/
    └── wandb/
```

### Poetry를 통한 환경 구축

먼저 Poetry로 프로젝트를 초기화하고 의존성을 설치한다.

```bash
# 프로젝트 초기화
poetry new llm-modeling
cd llm-modeling

# Python 3.11 가상환경 생성
poetry env use python3.11

# 의존성 설치
poetry add torch pytorch-lightning transformers datasets wandb
poetry add peft accelerate deepspeed bitsandbytes
poetry add --group dev jupyter black isort pytest

# 가상환경 활성화
poetry shell

# 패키지 설치 확인
poetry show
```

### 기본 설정 파일 생성

```python
# src/utils/config.py
"""프로젝트 전역 설정 관리"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class ModelConfig:
    """모델 관련 설정"""
    name: str = "gpt2"  # Hugging Face 모델 이름
    max_length: int = 512
    num_labels: Optional[int] = None
    cache_dir: Optional[str] = None

    # PEFT 설정
    use_peft: bool = False
    peft_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """데이터 관련 설정"""
    dataset_name: str = "imdb"
    train_split: str = "train"
    val_split: str = "test"
    test_split: Optional[str] = None

    batch_size: int = 8
    num_workers: int = 4
    max_samples: Optional[int] = None  # 디버깅용

    # 토크나이저 설정
    padding: str = "max_length"
    truncation: bool = True


@dataclass
class TrainingConfig:
    """훈련 관련 설정"""
    max_epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100

    # 분산 학습 설정
    accelerator: str = "gpu"
    devices: int = 1
    strategy: str = "auto"
    precision: str = "bf16-mixed"

    # 체크포인팅
    save_top_k: int = 3
    monitor: str = "val_loss"
    mode: str = "min"

    # 조기 종료
    patience: int = 3
    min_delta: float = 0.001


@dataclass
class WandbConfig:
    """Wandb 설정"""
    project: str = "llm-modeling"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    offline: bool = False


@dataclass
class Config:
    """전체 설정"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # 프로젝트 경로
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    def __post_init__(self):
        """설정 검증 및 후처리"""
        # 절대 경로로 변환
        self.output_dir = self.project_root / self.output_dir

        # 디렉토리 생성
        self.output_dir.mkdir(exist_ok=True, parents=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)

        # 환경 변수에서 Wandb 설정 읽기
        if not self.wandb.entity:
            self.wandb.entity = os.getenv("WANDB_ENTITY")


# 전역 설정 인스턴스
config = Config()

print("✅ 설정 파일 로드 완료!")
print(f"프로젝트 루트: {config.project_root}")
print(f"출력 디렉토리: {config.output_dir}")
```

## 🧠 LLM을 위한 기본 LightningModule

LLM 모델링의 기반이 되는 **BaseLLMModule**을 먼저 구현해보자. 이 클래스는 모든 LLM 태스크에 공통으로 사용되는 기능들을 제공한다.

```python
# src/models/base_llm.py
"""LLM 모델링을 위한 기본 Lightning 모듈"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    get_linear_schedule_with_warmup as transformers_scheduler
)
from peft import get_peft_model, LoraConfig, TaskType
from typing import Dict, Any, Optional, Tuple, Union
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseLLMModule(pl.LightningModule):
    """
    LLM 모델링을 위한 기본 Lightning 모듈

    이 클래스는 다음 기능들을 제공합니다:
    - Hugging Face 모델 로딩 및 설정
    - PEFT (LoRA/QLoRA) 지원
    - 옵티마이저 및 스케줄러 설정
    - 메트릭 로깅
    - 체크포인트 관리
    """

    def __init__(
            self,
            model_name: str = "gpt2",
            learning_rate: float = 2e-5,
            weight_decay: float = 0.01,
            warmup_steps: int = 100,
            max_epochs: int = 3,
            use_peft: bool = False,
            peft_config: Optional[Dict[str, Any]] = None,
            model_max_length: int = 512,
            **kwargs
    ):
        super().__init__()

        # 하이퍼파라미터 자동 저장 (Wandb에 자동 로깅됨)
        self.save_hyperparameters()

        # 모델 설정
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        self.use_peft = use_peft
        self.model_max_length = model_max_length

        # 모델과 토크나이저 초기화
        self._setup_model_and_tokenizer()

        # PEFT 적용 (LoRA 등)
        if use_peft:
            self._setup_peft(peft_config or {})

        # 메트릭 저장용 리스트
        self.training_step_outputs = []
        self.validation_step_outputs = []

        logger.info(f"✅ {self.__class__.__name__} 초기화 완료")
        logger.info(f"모델: {model_name}")
        logger.info(f"PEFT 사용: {use_peft}")
        logger.info(f"총 파라미터: {self.count_parameters():,}")

    def _setup_model_and_tokenizer(self):
        """모델과 토크나이저 설정"""
        try:
            # 설정 로드
            self.config = AutoConfig.from_pretrained(self.model_name)

            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # 패드 토큰 설정 (GPT 계열 모델의 경우)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 모델 최대 길이 설정
            if hasattr(self.tokenizer, 'model_max_length'):
                self.tokenizer.model_max_length = self.model_max_length

            # 기본 모델 로드 (하위 클래스에서 오버라이드)
            self.model = AutoModel.from_pretrained(
                self.model_name,
                config=self.config,
                torch_dtype=torch.bfloat16,  # 메모리 효율성
                attn_implementation="flash_attention_2",  # Flash Attention 사용
            )

            logger.info(f"모델과 토크나이저 로드 완료: {self.model_name}")

        except Exception as e:
            logger.error(f"모델 로딩 실패: {e}")
            raise

    def _setup_peft(self, peft_config: Dict[str, Any]):
        """PEFT (Parameter-Efficient Fine-Tuning) 설정"""
        default_peft_config = {
            "task_type": TaskType.CAUSAL_LM,  # 하위 클래스에서 오버라이드
            "r": 16,  # LoRA rank
            "lora_alpha": 32,  # LoRA alpha
            "lora_dropout": 0.1,  # LoRA dropout
            "target_modules": ["q_proj", "v_proj"],  # 적용할 모듈들
        }

        # 기본 설정과 사용자 설정 병합
        default_peft_config.update(peft_config)

        # LoRA 설정 생성
        lora_config = LoraConfig(**default_peft_config)

        # 모델에 PEFT 적용
        self.model = get_peft_model(self.model, lora_config)

        # 훈련 가능한 파라미터만 출력
        self.model.print_trainable_parameters()

        logger.info("PEFT (LoRA) 설정 완료")

    def count_parameters(self) -> int:
        """모델의 전체 파라미터 개수 계산"""
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        """훈련 가능한 파라미터 개수 계산"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, **inputs) -> Dict[str, torch.Tensor]:
        """
        순전파 - 하위 클래스에서 구체적으로 구현

        Args:
            **inputs: 모델 입력 (input_ids, attention_mask 등)

        Returns:
            모델 출력을 담은 딕셔너리
        """
        raise NotImplementedError("하위 클래스에서 구현해야 합니다.")

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        훈련 스텝 - 하위 클래스에서 구체적으로 구현

        Args:
            batch: 배치 데이터
            batch_idx: 배치 인덱스

        Returns:
            손실값
        """
        raise NotImplementedError("하위 클래스에서 구현해야 합니다.")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        검증 스텝 - 하위 클래스에서 구체적으로 구현

        Args:
            batch: 배치 데이터
            batch_idx: 배치 인덱스

        Returns:
            손실값
        """
        raise NotImplementedError("하위 클래스에서 구현해야 합니다.")

    def on_train_epoch_end(self):
        """훈련 에포크 종료 시 처리"""
        # 평균 메트릭 계산 및 로깅
        if self.training_step_outputs:
            avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
            self.log('train_loss_epoch', avg_loss, prog_bar=True, sync_dist=True)

            # GPU 메모리 사용량 로깅
            if torch.cuda.is_available():
                self.log('gpu_memory_gb', torch.cuda.memory_allocated() / 1024 ** 3)

            # 메모리 정리
            self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        """검증 에포크 종료 시 처리"""
        if self.validation_step_outputs:
            avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
            self.log('val_loss_epoch', avg_loss, prog_bar=True, sync_dist=True)

            # 메모리 정리
            self.validation_step_outputs.clear()

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        옵티마이저와 스케줄러 설정

        LLM 훈련에 최적화된 AdamW + Linear Warmup 스케줄러 사용
        """
        # 가중치 감쇠를 적용하지 않을 파라미터들
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

        # 파라미터 그룹 분리
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

        # AdamW 옵티마이저
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=1e-8,
            betas=(0.9, 0.999)  # BERT 논문 설정
        )

        # 총 스텝 수 계산
        total_steps = self.trainer.estimated_stepping_batches

        # Linear Warmup + Linear Decay 스케줄러
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )

        logger.info(f"옵티마이저 설정 완료")
        logger.info(f"학습률: {self.learning_rate}")
        logger.info(f"총 스텝: {total_steps}")
        logger.info(f"워밍업 스텝: {self.warmup_steps}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # 스텝마다 스케줄러 업데이트
                "frequency": 1,
                "name": "learning_rate"  # Wandb에 로깅될 이름
            }
        }

    def lr_scheduler_step(self, scheduler, metric):
        """학습률 스케줄러 스텝 (수동 제어)"""
        scheduler.step()

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환 (디버깅 및 로깅용)"""
        info = {
            "model_name": self.model_name,
            "total_parameters": self.count_parameters(),
            "trainable_parameters": self.count_trainable_parameters(),
            "use_peft": self.use_peft,
            "device": next(self.parameters()).device.type,
            "dtype": next(self.parameters()).dtype,
        }

        # PEFT 정보 추가
        if self.use_peft and hasattr(self.model, 'peft_config'):
            info["peft_config"] = self.model.peft_config

        return info


# 기본 모듈 테스트
if __name__ == "__main__":
    # 간단한 테스트
    module = BaseLLMModule(
        model_name="gpt2",
        use_peft=True,
        peft_config={"r": 8, "lora_alpha": 16}
    )

    print("모델 정보:")
    for key, value in module.get_model_info().items():
        print(f"  {key}: {value}")
```

## 📊 LLM 텍스트 분류 모듈 구현

이제 BaseLLMModule을 상속받아 **텍스트 분류 태스크**를 위한 구체적인 모듈을 구현해보자.

```python
# src/models/classification.py
"""텍스트 분류를 위한 Lightning 모듈"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForSequenceClassification
from torchmetrics import Accuracy, F1Score, Precision, Recall, AUROC
from typing import Dict, Any, Optional, List
import logging

from .base_llm import BaseLLMModule

logger = logging.getLogger(__name__)


class TextClassificationModule(BaseLLMModule):
    """
    텍스트 분류를 위한 Lightning 모듈

    Features:
    - 다중 클래스 분류 지원
    - 클래스 불균형 처리 (가중치 적용)
    - 다양한 메트릭 자동 계산
    - 혼동 행렬 로깅
    """

    def __init__(
            self,
            num_labels: int,
            class_weights: Optional[List[float]] = None,
            label_names: Optional[List[str]] = None,
            dropout_rate: float = 0.1,
            **kwargs
    ):
        # 분류 태스크용 설정 추가
        if 'peft_config' in kwargs and 'task_type' not in kwargs['peft_config']:
            from peft import TaskType
            kwargs['peft_config']['task_type'] = TaskType.SEQ_CLS

        super().__init__(**kwargs)

        self.num_labels = num_labels
        self.label_names = label_names or [f"Label_{i}" for i in range(num_labels)]
        self.dropout_rate = dropout_rate

        # 클래스 가중치 설정 (불균형 데이터 처리)
        self.register_buffer(
            'class_weights',
            torch.tensor(class_weights) if class_weights else None
        )

        # 모델 재설정 (분류용)
        self._setup_classification_model()

        # 손실 함수
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        # 메트릭 설정
        self._setup_metrics()

        logger.info(f"텍스트 분류 모듈 초기화 완료")
        logger.info(f"클래스 수: {num_labels}")
        logger.info(f"클래스 이름: {self.label_names}")

    def _setup_classification_model(self):
        """분류용 모델 설정"""
        # 기존 모델 제거
        del self.model

        # 분류용 모델 로드
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        # 분류기 헤드에 드롭아웃 추가
        if hasattr(self.model, 'classifier') and self.dropout_rate > 0:
            classifier_input_dim = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(classifier_input_dim, self.num_labels)
            )

        # PEFT 재적용 (필요한 경우)
        if self.use_peft:
            from peft import get_peft_model, LoraConfig, TaskType

            peft_config = getattr(self, 'hparams', {}).get('peft_config', {})
            default_config = {
                "task_type": TaskType.SEQ_CLS,
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"],
            }
            default_config.update(peft_config)

            lora_config = LoraConfig(**default_config)
            self.model = get_peft_model(self.model, lora_config)

    def _setup_metrics(self):
        """메트릭 설정"""
        task = "multiclass" if self.num_labels > 2 else "binary"
        average = "macro" if self.num_labels > 2 else "binary"

        # 훈련 메트릭
        self.train_accuracy = Accuracy(task=task, num_classes=self.num_labels)
        self.train_f1 = F1Score(task=task, num_classes=self.num_labels, average=average)

        # 검증 메트릭
        self.val_accuracy = Accuracy(task=task, num_classes=self.num_labels)
        self.val_f1 = F1Score(task=task, num_classes=self.num_labels, average=average)
        self.val_precision = Precision(task=task, num_classes=self.num_labels, average=average)
        self.val_recall = Recall(task=task, num_classes=self.num_labels, average=average)

        # AUROC (확률이 필요하므로 로짓 사용)
        if self.num_labels == 2:
            self.val_auroc = AUROC(task="binary")
        else:
            self.val_auroc = AUROC(task="multiclass", num_classes=self.num_labels)

        logger.info(f"메트릭 설정 완료: {task} 분류")

    def forward(self, input_ids: Tensor, attention_mask: Tensor, **kwargs) -> Dict[str, Tensor]:
        """
        순전파

        Args:
            input_ids: 토큰 ID 텐서 [batch_size, seq_len]
            attention_mask: 어텐션 마스크 [batch_size, seq_len]

        Returns:
            모델 출력 딕셔너리 (logits, hidden_states 등)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )

        return {
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states,
            "attentions": getattr(outputs, 'attentions', None)
        }

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """훈련 스텝"""
        # 순전파
        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        logits = outputs["logits"]
        labels = batch["labels"]

        # 손실 계산
        loss = self.criterion(logits, labels)

        # 예측 및 메트릭 계산
        preds = torch.argmax(logits, dim=-1)

        # 메트릭 업데이트
        self.train_accuracy(preds, labels)
        self.train_f1(preds, labels)

        # 로깅
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc", self.train_accuracy, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_f1", self.train_f1, on_epoch=True, sync_dist=True)

        # 학습률 로깅
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", current_lr, on_step=True, sync_dist=True)

        # 출력 저장 (에포크 종료 시 처리용)
        output = {
            "loss": loss,
            "preds": preds.detach(),
            "labels": labels.detach(),
            "logits": logits.detach()
        }
        self.training_step_outputs.append(output)

        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """검증 스텝"""
        # 순전파
        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        logits = outputs["logits"]
        labels = batch["labels"]

        # 손실 계산
        loss = self.criterion(logits, labels)

        # 예측 및 확률 계산
        preds = torch.argmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)

        # 메트릭 업데이트
        self.val_accuracy(preds, labels)
        self.val_f1(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)

        # AUROC 업데이트
        if self.num_labels == 2:
            self.val_auroc(probs[:, 1], labels)  # 양성 클래스 확률
        else:
            self.val_auroc(probs, labels)

        # 로깅
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_accuracy, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_f1", self.val_f1, on_epoch=True, sync_dist=True)
        self.log("val_precision", self.val_precision, on_epoch=True, sync_dist=True)
        self.log("val_recall", self.val_recall, on_epoch=True, sync_dist=True)
        self.log("val_auroc", self.val_auroc, on_epoch=True, sync_dist=True)

        # 출력 저장
        output = {
            "loss": loss,
            "preds": preds.detach(),
            "labels": labels.detach(),
            "probs": probs.detach()
        }
        self.validation_step_outputs.append(output)

        return loss

    def on_validation_epoch_end(self):
        """검증 에포크 종료 시 혼동 행렬 로깅"""
        super().on_validation_epoch_end()

        # 혼동 행렬 생성 및 로깅 (Wandb에 자동 전송)
        if self.validation_step_outputs and hasattr(self.logger, 'experiment'):
            all_preds = torch.cat([x["preds"] for x in self.validation_step_outputs])
            all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs])

            # Wandb confusion matrix
            try:
                import wandb

                # 혼동 행렬 생성
                cm = wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_labels.cpu().numpy(),
                    preds=all_preds.cpu().numpy(),
                    class_names=self.label_names
                )

                # Wandb에 로깅
                self.logger.experiment.log({"confusion_matrix": cm})

            except ImportError:
                logger.warning("Wandb를 사용할 수 없어 혼동 행렬 로깅을 건너뜁니다.")

    def predict_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        """예측 스텝 (추론용)"""
        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        logits = outputs["logits"]
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)

        return {
            "predictions": preds,
            "probabilities": probs,
            "logits": logits
        }

    def get_sample_predictions(self, texts: List[str], max_samples: int = 5) -> List[Dict[str, Any]]:
        """
        샘플 텍스트에 대한 예측 결과 반환 (디버깅용)

        Args:
            texts: 예측할 텍스트 리스트
            max_samples: 최대 샘플 수

        Returns:
            예측 결과 리스트
        """
        self.eval()
        results = []

        with torch.no_grad():
            for i, text in enumerate(texts[:max_samples]):
                # 토크나이징
                inputs = self.tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    max_length=self.model_max_length,
                    return_tensors="pt"
                ).to(self.device)

                # 예측
                outputs = self.forward(**inputs)
                logits = outputs["logits"]
                probs = F.softmax(logits, dim=-1)
                pred_idx = torch.argmax(logits, dim=-1).item()
                confidence = probs[0, pred_idx].item()

                results.append({
                    "text": text,
                    "predicted_label": self.label_names[pred_idx],
                    "confidence": confidence,
                    "all_probabilities": {
                        self.label_names[j]: probs[0, j].item()
                        for j in range(self.num_labels)
                    }
                })

        return results


# 사용 예시
if __name__ == "__main__":
    # 이진 분류 모델 테스트
    model = TextClassificationModule(
        model_name="distilbert-base-uncased",
        num_labels=2,
        label_names=["Negative", "Positive"],
        learning_rate=2e-5,
        use_peft=True,
        peft_config={"r": 8, "lora_alpha": 16}
    )

    print("분류 모델 정보:")
    for key, value in model.get_model_info().items():
        print(f"  {key}: {value}")

    # 샘플 예측 테스트
    sample_texts = [
        "This movie is amazing!",
        "I hate this film.",
        "It's okay, not great but not terrible either."
    ]

    predictions = model.get_sample_predictions(sample_texts)
    print("\n샘플 예측:")
    for pred in predictions:
        print(f"Text: {pred['text']}")
        print(f"Prediction: {pred['predicted_label']} (confidence: {pred['confidence']:.3f})")
        print()
```

## 📝 텍스트 생성 모듈 구현

다음으로 **텍스트 생성 태스크**를 위한 모듈을 구현해보자. 이 모듈은 언어 모델링과 조건부 생성을 지원한다.

```python
# src/models/generation.py
"""텍스트 생성을 위한 Lightning 모듈"""

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM, GenerationConfig
from torchmetrics.text import Perplexity, BLEUScore, ROUGEScore
from typing import Dict, Any, Optional, List, Union
import logging

from .base_llm import BaseLLMModule

logger = logging.getLogger(__name__)


class TextGenerationModule(BaseLLMModule):
    """
    텍스트 생성을 위한 Lightning 모듈

    Features:
    - Causal Language Modeling
    - 조건부 텍스트 생성
    - 다양한 디코딩 전략 지원
    - 생성 품질 메트릭 자동 계산
    """

    def __init__(
            self,
            max_new_tokens: int = 50,
            temperature: float = 1.0,
            top_k: int = 50,
            top_p: float = 0.9,
            do_sample: bool = True,
            pad_token_id: Optional[int] = None,
            **kwargs
    ):
        # 생성 태스크용 설정
        if 'peft_config' in kwargs and 'task_type' not in kwargs['peft_config']:
            from peft import TaskType
            kwargs['peft_config']['task_type'] = TaskType.CAUSAL_LM

        super().__init__(**kwargs)

        # 생성 설정
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample

        # 모델 재설정 (생성용)
        self._setup_generation_model()

        # 생성 설정
        self._setup_generation_config(pad_token_id)

        # 메트릭 설정
        self._setup_metrics()

        logger.info(f"텍스트 생성 모듈 초기화 완료")
        logger.info(f"최대 생성 토큰: {max_new_tokens}")
        logger.info(f"샘플링: {do_sample}, 온도: {temperature}")

    def _setup_generation_model(self):
        """생성용 모델 설정"""
        # 기존 모델 제거
        del self.model

        # 생성용 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        # PEFT 재적용 (필요한 경우)
        if self.use_peft:
            from peft import get_peft_model, LoraConfig, TaskType

            peft_config = getattr(self, 'hparams', {}).get('peft_config', {})
            default_config = {
                "task_type": TaskType.CAUSAL_LM,
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            }
            default_config.update(peft_config)

            lora_config = LoraConfig(**default_config)
            self.model = get_peft_model(self.model, lora_config)

    def _setup_generation_config(self, pad_token_id: Optional[int]):
        """생성 설정"""
        # 패드 토큰 설정
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            do_sample=self.do_sample,
            pad_token_id=pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1,  # 반복 방지
            length_penalty=1.0,  # 길이 페널티
        )

        # 모델에 생성 설정 적용
        self.model.generation_config = self.generation_config

    def _setup_metrics(self):
        """메트릭 설정"""
        # Perplexity (혼란도) - 언어 모델 품질 지표
        self.train_perplexity = Perplexity()
        self.val_perplexity = Perplexity()

        # BLEU Score - 생성 품질 지표 (참조 텍스트 필요)
        self.val_bleu = BLEUScore()

        # ROUGE Score - 요약 품질 지표
        self.val_rouge = ROUGEScore()

        logger.info("생성 메트릭 설정 완료")

    def forward(self, input_ids: Tensor, attention_mask: Tensor, **kwargs) -> Dict[str, Tensor]:
        """
        순전파

        Args:
            input_ids: 토큰 ID 텐서 [batch_size, seq_len]
            attention_mask: 어텐션 마스크 [batch_size, seq_len]

        Returns:
            모델 출력 딕셔너리
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,  # Causal LM은 input과 label이 동일
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )

        return {
            "logits": outputs.logits,
            "loss": outputs.loss,
            "hidden_states": outputs.hidden_states,
            "attentions": getattr(outputs, 'attentions', None)
        }

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """훈련 스텝"""
        # 순전파
        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        loss = outputs["loss"]
        logits = outputs["logits"]

        # Perplexity 계산
        # 마지막 토큰을 제외한 logits와 첫 번째 토큰을 제외한 labels 사용
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["input_ids"][..., 1:].contiguous()

        # 패딩 토큰 마스킹
        attention_mask = batch["attention_mask"][..., 1:].contiguous()

        # Perplexity 업데이트 (유효한 토큰만)
        valid_mask = attention_mask.bool()
        if valid_mask.sum() > 0:
            valid_logits = shift_logits[valid_mask]
            valid_labels = shift_labels[valid_mask]
            self.train_perplexity(valid_logits, valid_labels)

        # 로깅
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_perplexity", self.train_perplexity, on_epoch=True, prog_bar=True, sync_dist=True)

        # 학습률 로깅
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", current_lr, on_step=True, sync_dist=True)

        # 출력 저장
        output = {
            "loss": loss,
            "logits": logits.detach(),
            "labels": batch["input_ids"].detach()
        }
        self.training_step_outputs.append(output)

        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """검증 스텝"""
        # 순전파
        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        loss = outputs["loss"]
        logits = outputs["logits"]

        # Perplexity 계산
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["input_ids"][..., 1:].contiguous()
        attention_mask = batch["attention_mask"][..., 1:].contiguous()

        valid_mask = attention_mask.bool()
        if valid_mask.sum() > 0:
            valid_logits = shift_logits[valid_mask]
            valid_labels = shift_labels[valid_mask]
            self.val_perplexity(valid_logits, valid_labels)

        # 로깅
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_perplexity", self.val_perplexity, on_epoch=True, prog_bar=True, sync_dist=True)

        # 출력 저장
        output = {
            "loss": loss,
            "input_ids": batch["input_ids"].detach(),
            "attention_mask": batch["attention_mask"].detach()
        }
        self.validation_step_outputs.append(output)

        return loss

    def generate_text(
            self,
            prompt: str,
            max_new_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            do_sample: Optional[bool] = None,
            num_return_sequences: int = 1
    ) -> List[str]:
        """
        텍스트 생성

        Args:
            prompt: 생성 시작 프롬프트
            max_new_tokens: 최대 생성 토큰 수
            temperature: 생성 온도
            top_k: Top-k 샘플링
            top_p: Top-p (nucleus) 샘플링
            do_sample: 샘플링 여부
            num_return_sequences: 반환할 시퀀스 수

        Returns:
            생성된 텍스트 리스트
        """
        self.eval()

        # 입력 토크나이징
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_max_length - (max_new_tokens or self.max_new_tokens)
        ).to(self.device)

        # 생성 설정 오버라이드
        generation_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "temperature": temperature or self.temperature,
            "top_k": top_k or self.top_k,
            "top_p": top_p or self.top_p,
            "do_sample": do_sample if do_sample is not None else self.do_sample,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        with torch.no_grad():
            # 생성
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs
            )

            # 디코딩
            generated_texts = []
            for output in outputs:
                # 원래 입력 제거
                new_tokens = output[inputs["input_ids"].shape[1]:]
                generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                generated_texts.append(generated_text)

        return generated_texts

    def on_validation_epoch_end(self):
        """검증 에포크 종료 시 샘플 생성"""
        super().on_validation_epoch_end()

        # 샘플 생성 및 로깅 (Wandb에 자동 전송)
        if hasattr(self.logger, 'experiment'):
            sample_prompts = [
                "The future of artificial intelligence is",
                "In a world where technology",
                "Once upon a time in a distant galaxy"
            ]

            try:
                import wandb

                # 각 프롬프트에 대해 생성
                generated_samples = []
                for prompt in sample_prompts:
                    generated_texts = self.generate_text(
                        prompt=prompt,
                        max_new_tokens=50,
                        temperature=0.8,
                        num_return_sequences=2
                    )

                    for i, text in enumerate(generated_texts):
                        generated_samples.append([prompt, f"Sample {i + 1}", text])

                # Wandb 테이블로 로깅
                table = wandb.Table(
                    columns=["Prompt", "Sample", "Generated Text"],
                    data=generated_samples
                )

                self.logger.experiment.log({"generated_samples": table})

            except ImportError:
                logger.warning("Wandb를 사용할 수 없어 생성 샘플 로깅을 건너뜁니다.")

    def predict_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Any]:
        """예측 스텝 (추론용)"""
        # 입력 텍스트 디코딩
        prompts = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in batch["input_ids"]
        ]

        # 각 프롬프트에 대해 생성
        all_generated = []
        for prompt in prompts:
            generated = self.generate_text(prompt, num_return_sequences=1)
            all_generated.extend(generated)

        return {
            "prompts": prompts,
            "generated_texts": all_generated
        }


# 사용 예시
if __name__ == "__main__":
    # 생성 모델 테스트
    model = TextGenerationModule(
        model_name="gpt2",
        max_new_tokens=50,
        temperature=0.8,
        learning_rate=1e-4,
        use_peft=True,
        peft_config={"r": 8, "lora_alpha": 16}
    )

    print("생성 모델 정보:")
    for key, value in model.get_model_info().items():
        print(f"  {key}: {value}")

    # 샘플 생성 테스트
    sample_prompts = [
        "The future of AI is",
        "In the year 2050,",
        "Machine learning will"
    ]

    print("\n샘플 생성:")
    for prompt in sample_prompts:
        generated = model.generate_text(prompt, max_new_tokens=30, temperature=0.8)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated[0]}")
        print()
```

## 💾 LLM 데이터 모듈 구현

LLM 학습을 위한 **효율적인 데이터 처리**를 담당하는 DataModule을 구현해보자.

```python
# src/data/base_datamodule.py
"""LLM 학습을 위한 기본 데이터 모듈"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset as HFDataset
from typing import Optional, Dict, Any, List, Union, Callable
import torch
import logging

logger = logging.getLogger(__name__)


class LLMDataModule(pl.LightningDataModule):
    """
    LLM 학습을 위한 기본 데이터 모듈

    Features:
    - Hugging Face Datasets 통합
    - 자동 토크나이징
    - 동적 패딩
    - 메모리 효율적 처리
    """

    def __init__(
            self,
            tokenizer_name: str,
            dataset_name: Optional[str] = None,
            dataset_config: Optional[str] = None,
            train_split: str = "train",
            val_split: str = "validation",
            test_split: Optional[str] = "test",
            batch_size: int = 8,
            max_length: int = 512,
            num_workers: int = 4,
            pin_memory: bool = True,
            train_val_split: float = 0.9,
            streaming: bool = False,
            cache_dir: Optional[str] = None,
            max_samples: Optional[int] = None,  # 디버깅용
            **tokenizer_kwargs
    ):
        super().__init__()

        # 설정 저장
        self.tokenizer_name = tokenizer_name
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_val_split = train_val_split
        self.streaming = streaming
        self.cache_dir = cache_dir
        self.max_samples = max_samples
        self.tokenizer_kwargs = tokenizer_kwargs

        # 토크나이저 로드
        self._setup_tokenizer()

        # 데이터셋 변수 초기화
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        logger.info(f"LLM 데이터 모듈 초기화 완료")
        logger.info(f"토크나이저: {tokenizer_name}")
        logger.info(f"데이터셋: {dataset_name}")
        logger.info(f"배치 크기: {batch_size}")
        logger.info(f"최대 길이: {max_length}")

    def _setup_tokenizer(self):
        """토크나이저 설정"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            cache_dir=self.cache_dir,
            **self.tokenizer_kwargs
        )

        # 패드 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("패드 토큰을 EOS 토큰으로 설정했습니다.")

        # 모델 최대 길이 설정
        if hasattr(self.tokenizer, 'model_max_length'):
            self.tokenizer.model_max_length = self.max_length

        logger.info(f"토크나이저 설정 완료: {self.tokenizer_name}")
        logger.info(f"어휘 크기: {len(self.tokenizer)}")
        logger.info(f"패드 토큰: {self.tokenizer.pad_token}")

    def prepare_data(self):
        """데이터 다운로드 (멀티 프로세스 환경에서 한 번만 실행)"""
        if self.dataset_name:
            try:
                load_dataset(
                    self.dataset_name,
                    self.dataset_config,
                    cache_dir=self.cache_dir,
                    streaming=self.streaming
                )
                logger.info(f"데이터셋 다운로드 완료: {self.dataset_name}")
            except Exception as e:
                logger.error(f"데이터셋 다운로드 실패: {e}")
                raise

    def setup(self, stage: Optional[str] = None):
        """데이터셋 설정 (각 프로세스에서 실행)"""
        if self.dataset_name is None:
            logger.warning("데이터셋이 지정되지 않았습니다. 커스텀 데이터를 사용하세요.")
            return

        try:
            # 데이터셋 로드
            dataset = load_dataset(
                self.dataset_name,
                self.dataset_config,
                cache_dir=self.cache_dir,
                streaming=self.streaming
            )

            if stage == "fit" or stage is None:
                # 훈련/검증 데이터 설정
                if self.val_split in dataset:
                    train_data = dataset[self.train_split]
                    val_data = dataset[self.val_split]
                else:
                    # 검증 데이터가 없으면 훈련 데이터에서 분할
                    full_train = dataset[self.train_split]

                    if self.streaming:
                        # 스트리밍 데이터는 분할하지 않고 그대로 사용
                        train_data = full_train
                        val_data = full_train  # 동일한 데이터 사용 (주의: 실제로는 별도 데이터 필요)
                        logger.warning("스트리밍 모드에서는 적절한 검증 데이터 분할이 어렵습니다.")
                    else:
                        # 메모리에 로드하여 분할
                        total_size = len(full_train)
                        train_size = int(total_size * self.train_val_split)
                        val_size = total_size - train_size

                        train_data, val_data = random_split(
                            full_train, [train_size, val_size]
                        )

                # 샘플 수 제한 (디버깅용)
                if self.max_samples and not self.streaming:
                    train_data = train_data.select(range(min(self.max_samples, len(train_data))))
                    val_data = val_data.select(range(min(self.max_samples // 5, len(val_data))))

                # 토크나이징 적용
                self.train_dataset = self._tokenize_dataset(train_data, "train")
                self.val_dataset = self._tokenize_dataset(val_data, "validation")

                logger.info(
                    f"훈련 데이터 크기: {len(self.train_dataset) if hasattr(self.train_dataset, '__len__') else 'Unknown (streaming)'}")
                logger.info(
                    f"검증 데이터 크기: {len(self.val_dataset) if hasattr(self.val_dataset, '__len__') else 'Unknown (streaming)'}")

            if stage == "test" or stage is None:
                # 테스트 데이터 설정
                if self.test_split and self.test_split in dataset:
                    test_data = dataset[self.test_split]

                    if self.max_samples and not self.streaming:
                        test_data = test_data.select(range(min(self.max_samples // 5, len(test_data))))

                    self.test_dataset = self._tokenize_dataset(test_data, "test")
                    logger.info(
                        f"테스트 데이터 크기: {len(self.test_dataset) if hasattr(self.test_dataset, '__len__') else 'Unknown (streaming)'}")

        except Exception as e:
            logger.error(f"데이터셋 설정 실패: {e}")
            raise

    def _tokenize_dataset(self, dataset: Union[HFDataset, Dataset], split: str) -> HFDataset:
        """
        데이터셋 토크나이징 - 하위 클래스에서 오버라이드

        Args:
            dataset: 원본 데이터셋
            split: 데이터 분할 이름 ("train", "validation", "test")

        Returns:
            토크나이징된 데이터셋
        """
        raise NotImplementedError("하위 클래스에서 구현해야 합니다.")

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        배치 콜레이션 함수 - 동적 패딩 적용

        Args:
            batch: 배치 데이터 리스트

        Returns:
            패딩된 배치 텐서
        """
        # 배치에서 키 추출
        keys = batch[0].keys()

        # 텐서로 변환
        batch_dict = {}
        for key in keys:
            if key in ["input_ids", "attention_mask", "labels"]:
                # 시퀀스 데이터는 패딩 적용
                sequences = [item[key] for item in batch]

                # 패딩
                if isinstance(sequences[0], torch.Tensor):
                    batch_dict[key] = torch.nn.utils.rnn.pad_sequence(
                        sequences, batch_first=True, padding_value=self.tokenizer.pad_token_id
                    )
                else:
                    # 토크나이저를 사용한 패딩
                    batch_dict[key] = self.tokenizer.pad(
                        {"input_ids": sequences},
                        padding=True,
                        return_tensors="pt"
                    )["input_ids"]
            else:
                # 다른 데이터는 스택
                batch_dict[key] = torch.stack([item[key] for item in batch])

        return batch_dict

    def train_dataloader(self) -> DataLoader:
        """훈련 데이터로더"""
        if self.train_dataset is None:
            raise ValueError("훈련 데이터셋이 설정되지 않았습니다. setup()을 먼저 호출하세요.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # 스트리밍 모드에서는 자동으로 False
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self) -> DataLoader:
        """검증 데이터로더"""
        if self.val_dataset is None:
            raise ValueError("검증 데이터셋이 설정되지 않았습니다. setup()을 먼저 호출하세요.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self) -> DataLoader:
        """테스트 데이터로더"""
        if self.test_dataset is None:
            raise ValueError("테스트 데이터셋이 설정되지 않았습니다.")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )

    def get_sample_data(self, split: str = "train", num_samples: int = 3) -> List[Dict[str, Any]]:
        """
        샘플 데이터 반환 (디버깅용)

        Args:
            split: 데이터 분할 ("train", "validation", "test")
            num_samples: 샘플 수

        Returns:
            샘플 데이터 리스트
        """
        if split == "train" and self.train_dataset:
            dataset = self.train_dataset
        elif split == "validation" and self.val_dataset:
            dataset = self.val_dataset
        elif split == "test" and self.test_dataset:
            dataset = self.test_dataset
        else:
            raise ValueError(f"'{split}' 데이터셋을 찾을 수 없습니다.")

        samples = []
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]

            # 토큰을 텍스트로 디코딩
            if "input_ids" in sample:
                sample["decoded_text"] = self.tokenizer.decode(
                    sample["input_ids"],
                    skip_special_tokens=True
                )

            samples.append(sample)

        return samples


# 텍스트 분류를 위한 구체적인 데이터 모듈
class TextClassificationDataModule(LLMDataModule):
    """텍스트 분류를 위한 데이터 모듈"""

    def __init__(
            self,
            text_column: str = "text",
            label_column: str = "label",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.text_column = text_column
        self.label_column = label_column

    def _tokenize_dataset(self, dataset: HFDataset, split: str) -> HFDataset:
        """텍스트 분류용 토크나이징"""

        def tokenize_function(examples):
            # 텍스트 토크나이징
            tokenized = self.tokenizer(
                examples[self.text_column],
                truncation=True,
                padding=False,  # 배치에서 동적 패딩
                max_length=self.max_length,
                return_tensors=None  # 리스트로 반환
            )

            # 레이블 추가
            if self.label_column in examples:
                tokenized["labels"] = examples[self.label_column]

            return tokenized

        # 토크나이징 적용
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names if hasattr(dataset, 'column_names') else []
        )

        return tokenized_dataset


# 텍스트 생성을 위한 구체적인 데이터 모듈
class TextGenerationDataModule(LLMDataModule):
    """텍스트 생성을 위한 데이터 모듈"""

    def __init__(
            self,
            text_column: str = "text",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.text_column = text_column

    def _tokenize_dataset(self, dataset: HFDataset, split: str) -> HFDataset:
        """텍스트 생성용 토크나이징"""

        def tokenize_function(examples):
            # 텍스트 토크나이징
            tokenized = self.tokenizer(
                examples[self.text_column],
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_tensors=None
            )

            # Causal LM에서는 input_ids와 labels가 동일
            tokenized["labels"] = tokenized["input_ids"].copy()

            return tokenized

        # 토크나이징 적용
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names if hasattr(dataset, 'column_names') else []
        )

        return tokenized_dataset


# 사용 예시
if __name__ == "__main__":
    # 분류 데이터 모듈 테스트
    data_module = TextClassificationDataModule(
        tokenizer_name="distilbert-base-uncased",
        dataset_name="imdb",
        batch_size=8,
        max_length=256,
        max_samples=100  # 테스트용
    )

    # 데이터 준비
    data_module.prepare_data()
    data_module.setup("fit")

    # 샘플 데이터 확인
    samples = data_module.get_sample_data("train", num_samples=2)
    for i, sample in enumerate(samples):
        print(f"샘플 {i + 1}:")
        print(f"  텍스트: {sample['decoded_text'][:100]}...")
        print(f"  레이블: {sample['labels']}")
        print()

    # 데이터로더 테스트
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))

    print("배치 정보:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
```

## 🔧 커스텀 콜백 구현

LLM 학습을 효과적으로 모니터링하고 제어하기 위한 **커스텀 콜백들**을 구현해보자.

```python
# src/callbacks/model_monitoring.py
"""LLM 모델 모니터링을 위한 커스텀 콜백들"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Optional, Union
import logging
import time
import psutil
import GPUtil

logger = logging.getLogger(__name__)


class LLMModelMonitoringCallback(Callback):
    """
    LLM 모델 성능과 리소스 사용량을 모니터링하는 콜백

    Features:
    - GPU/CPU 메모리 사용량 추적
    - 그래디언트 노름 모니터링
    - 가중치 분포 추적
    - 학습 속도 측정
    """

    def __init__(
            self,
            log_every_n_steps: int = 100,
            monitor_gradients: bool = True,
            monitor_weights: bool = True,
            monitor_resources: bool = True,
            gradient_clip_threshold: float = 1.0
    ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.monitor_gradients = monitor_gradients
        self.monitor_weights = monitor_weights
        self.monitor_resources = monitor_resources
        self.gradient_clip_threshold = gradient_clip_threshold

        # 시간 추적
        self.batch_start_time = None
        self.epoch_start_time = None

        # 메트릭 누적
        self.step_times = []
        self.gradient_norms = []

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """에포크 시작 시 시간 기록"""
        self.epoch_start_time = time.time()
        self.step_times = []
        self.gradient_norms = []

        logger.info(f"에포크 {trainer.current_epoch + 1} 시작")

    def on_train_batch_start(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            batch: Any,
            batch_idx: int
    ) -> None:
        """배치 시작 시 시간 기록"""
        self.batch_start_time = time.time()

    def on_train_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Any,
            batch: Any,
            batch_idx: int
    ) -> None:
        """배치 종료 시 모니터링"""
        # 배치 처리 시간 계산
        if self.batch_start_time:
            batch_time = time.time() - self.batch_start_time
            self.step_times.append(batch_time)

        # 주기적 로깅
        if batch_idx % self.log_every_n_steps == 0:
            self._log_monitoring_metrics(trainer, pl_module, batch_idx)

    def on_after_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """백프롭 후 그래디언트 모니터링"""
        if self.monitor_gradients:
            self._monitor_gradients(trainer, pl_module)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """에포크 종료 시 요약 로깅"""
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time

            # 평균 배치 시간
            avg_batch_time = np.mean(self.step_times) if self.step_times else 0

            # 로깅
            pl_module.log("epoch_time_minutes", epoch_time / 60, sync_dist=True)
            pl_module.log("avg_batch_time_seconds", avg_batch_time, sync_dist=True)

            # 그래디언트 노름 통계
            if self.gradient_norms:
                pl_module.log("avg_gradient_norm", np.mean(self.gradient_norms), sync_dist=True)
                pl_module.log("max_gradient_norm", np.max(self.gradient_norms), sync_dist=True)

            logger.info(f"에포크 {trainer.current_epoch + 1} 완료 - 소요 시간: {epoch_time / 60:.2f}분")

    def _log_monitoring_metrics(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            batch_idx: int
    ) -> None:
        """모니터링 메트릭 로깅"""
        metrics = {}

        # 리소스 사용량 모니터링
        if self.monitor_resources:
            # GPU 메모리
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 ** 3  # GB
                gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 ** 3  # GB
                gpu_memory_free = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
                gpu_memory_free = gpu_memory_free / 1024 ** 3  # GB

                metrics.update({
                    "gpu_memory_allocated_gb": gpu_memory_allocated,
                    "gpu_memory_reserved_gb": gpu_memory_reserved,
                    "gpu_memory_free_gb": gpu_memory_free,
                })

                # GPU 사용률 (GPUtil 사용)
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # 첫 번째 GPU
                        metrics.update({
                            "gpu_utilization_percent": gpu.load * 100,
                            "gpu_temperature_celsius": gpu.temperature,
                        })
                except:
                    pass

            # CPU 메모리
            cpu_memory = psutil.virtual_memory()
            metrics.update({
                "cpu_memory_used_gb": cpu_memory.used / 1024 ** 3,
                "cpu_memory_percent": cpu_memory.percent,
                "cpu_utilization_percent": psutil.cpu_percent(),
            })

        # 가중치 모니터링
        if self.monitor_weights:
            weight_stats = self._get_weight_statistics(pl_module)
            metrics.update(weight_stats)

        # 배치 처리 속도
        if self.step_times:
            recent_times = self.step_times[-self.log_every_n_steps:]
            metrics["recent_avg_batch_time"] = np.mean(recent_times)
            metrics["throughput_samples_per_second"] = trainer.datamodule.batch_size / np.mean(recent_times)

        # 로깅
        for key, value in metrics.items():
            pl_module.log(key, value, on_step=True, sync_dist=True)

    def _monitor_gradients(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """그래디언트 모니터링"""
        total_norm = 0.0
        param_count = 0

        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            self.gradient_norms.append(total_norm)

            # 그래디언트 클리핑 경고
            if total_norm > self.gradient_clip_threshold:
                logger.warning(f"높은 그래디언트 노름 감지: {total_norm:.4f}")

            # 주기적 로깅
            if trainer.global_step % self.log_every_n_steps == 0:
                pl_module.log("gradient_norm", total_norm, on_step=True, sync_dist=True)

    def _get_weight_statistics(self, pl_module: pl.LightningModule) -> Dict[str, float]:
        """가중치 통계 계산"""
        stats = {}

        # 전체 가중치 통계
        all_weights = []
        for name, param in pl_module.named_parameters():
            if param.requires_grad and 'weight' in name:
                all_weights.append(param.data.flatten())

        if all_weights:
            all_weights = torch.cat(all_weights)
            stats.update({
                "weight_mean": all_weights.mean().item(),
                "weight_std": all_weights.std().item(),
                "weight_min": all_weights.min().item(),
                "weight_max": all_weights.max().item(),
            })

        # 레이어별 통계 (주요 레이어만)
        for name, param in pl_module.named_parameters():
            if param.requires_grad and any(layer in name for layer in ['embed', 'lm_head', 'classifier']):
                layer_name = name.split('.')[0]  # 첫 번째 부분만 사용
                stats[f"{layer_name}_weight_norm"] = param.data.norm().item()

        return stats


class TextGenerationCallback(Callback):
    """
    텍스트 생성 모델을 위한 콜백

    Features:
    - 주기적 샘플 생성
    - 생성 품질 모니터링
    - 토큰 분포 분석
    """

    def __init__(
            self,
            sample_prompts: List[str],
            generate_every_n_epochs: int = 1,
            max_new_tokens: int = 50,
            num_samples_per_prompt: int = 2,
            temperature: float = 0.8,
            top_k: int = 50,
            top_p: float = 0.9
    ):
        super().__init__()
        self.sample_prompts = sample_prompts
        self.generate_every_n_epochs = generate_every_n_epochs
        self.max_new_tokens = max_new_tokens
        self.num_samples_per_prompt = num_samples_per_prompt
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """검증 에포크 종료 시 샘플 생성"""
        if (trainer.current_epoch + 1) % self.generate_every_n_epochs == 0:
            self._generate_and_log_samples(trainer, pl_module)

    def _generate_and_log_samples(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """샘플 생성 및 로깅"""
        if not hasattr(pl_module, 'generate_text'):
            logger.warning("모델에 generate_text 메서드가 없습니다.")
            return

        pl_module.eval()

        try:
            import wandb

            # 각 프롬프트에 대해 생성
            all_samples = []
            for prompt in self.sample_prompts:
                generated_texts = pl_module.generate_text(
                    prompt=prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    num_return_sequences=self.num_samples_per_prompt
                )

                for i, text in enumerate(generated_texts):
                    all_samples.append([
                        f"Epoch {trainer.current_epoch + 1}",
                        prompt,
                        f"Sample {i + 1}",
                        text
                    ])

            # Wandb 테이블로 로깅
            if hasattr(trainer.logger, 'experiment'):
                table = wandb.Table(
                    columns=["Epoch", "Prompt", "Sample", "Generated Text"],
                    data=all_samples
                )
                trainer.logger.experiment.log({
                    f"generated_samples_epoch_{trainer.current_epoch + 1}": table
                })

            logger.info(f"에포크 {trainer.current_epoch + 1}: {len(all_samples)}개 샘플 생성 완료")

        except ImportError:
            logger.warning("Wandb를 사용할 수 없어 생성 샘플 로깅을 건너뜁니다.")
        except Exception as e:
            logger.error(f"샘플 생성 중 오류 발생: {e}")

        pl_module.train()


class ModelSizeCallback(Callback):
    """
    모델 크기와 파라미터 정보를 로깅하는 콜백
    """

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """학습 시작 시 모델 정보 로깅"""
        # 모델 크기 정보
        total_params = sum(p.numel() for p in pl_module.parameters())
        trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        # 메모리 사용량 (대략적 계산)
        model_size_mb = total_params * 4 / 1024 / 1024  # float32 기준

        # 로깅
        metrics = {
            "model_total_parameters": total_params,
            "model_trainable_parameters": trainable_params,
            "model_frozen_parameters": frozen_params,
            "model_size_mb": model_size_mb,
            "trainable_parameter_ratio": trainable_params / total_params if total_params > 0 else 0
        }

        for key, value in metrics.items():
            pl_module.log(key, value)

        # 콘솔 출력
        logger.info("=" * 50)
        logger.info("모델 정보")
        logger.info("=" * 50)
        logger.info(f"총 파라미터: {total_params:,}")
        logger.info(f"훈련 가능한 파라미터: {trainable_params:,}")
        logger.info(f"고정된 파라미터: {frozen_params:,}")
        logger.info(f"모델 크기: {model_size_mb:.2f} MB")
        logger.info(f"훈련 가능 비율: {trainable_params / total_params * 100:.2f}%")
        logger.info("=" * 50)

        # PEFT 정보 (사용 중인 경우)
        if hasattr(pl_module, 'use_peft') and pl_module.use_peft:
            if hasattr(pl_module.model, 'print_trainable_parameters'):
                logger.info("PEFT 파라미터 정보:")
                pl_module.model.print_trainable_parameters()


# 사용 예시
if __name__ == "__main__":
    # 콜백 테스트
    monitoring_callback = LLMModelMonitoringCallback(
        log_every_n_steps=50,
        monitor_gradients=True,
        monitor_weights=True,
        monitor_resources=True
    )

    generation_callback = TextGenerationCallback(
        sample_prompts=[
            "The future of AI is",
            "In a world where technology",
            "Once upon a time"
        ],
        generate_every_n_epochs=2,
        max_new_tokens=30
    )

    model_size_callback = ModelSizeCallback()

    print("커스텀 콜백들이 생성되었습니다:")
    print(f"- 모니터링 콜백: {monitoring_callback.__class__.__name__}")
    print(f"- 생성 콜백: {generation_callback.__class__.__name__}")
    print(f"- 모델 크기 콜백: {model_size_callback.__class__.__name__}")
```

## 📊 Wandb Logger 통합과 실험 추적

LLM 실험을 체계적으로 관리하기 위한 **Wandb 통합**을 구현해보자.

```python
# src/utils/wandb_integration.py
"""Wandb를 활용한 실험 추적 및 관리"""

import os
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from typing import Dict, Any, Optional, List, Union
import torch
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EnhancedWandbLogger(WandbLogger):
    """
    Wandb Logger 확장 클래스

    Features:
    - 하이퍼파라미터 자동 로깅
    - 모델 아티팩트 관리
    - 코드 스냅샷 저장
    - 환경 정보 기록
    """

    def __init__(
            self,
            project: str,
            entity: Optional[str] = None,
            name: Optional[str] = None,
            tags: Optional[List[str]] = None,
            notes: Optional[str] = None,
            config: Optional[Dict[str, Any]] = None,
            save_code: bool = True,
            save_artifacts: bool = True,
            **kwargs
    ):
        super().__init__(
            project=project,
            entity=entity,
            name=name,
            tags=tags,
            notes=notes,
            **kwargs
        )

        self.save_code = save_code
        self.save_artifacts = save_artifacts
        self._config = config or {}

        # 환경 정보 수집
        self._log_environment_info()

        logger.info(f"Enhanced Wandb Logger 초기화 완료")
        logger.info(f"프로젝트: {project}")
        logger.info(f"실행 이름: {name or 'auto-generated'}")

    def _log_environment_info(self):
        """환경 정보 로깅"""
        env_info = {
            "python_version": ".".join(map(str, __import__("sys").version_info[:3])),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_names": [torch.cuda.get_device_name(i) for i in
                          range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        }

        # GPU 메모리 정보
        if torch.cuda.is_available():
            gpu_memory_info = []
            for i in range(torch.cuda.device_count()):
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
                gpu_memory_info.append(f"GPU {i}: {memory_total:.1f} GB")
            env_info["gpu_memory"] = gpu_memory_info

        # 설정에 추가
        self._config.update({"environment": env_info})

    @property
    def experiment(self) -> wandb.sdk.wandb_run.Run:
        """Wandb run 객체 반환"""
        if self._experiment is None:
            self._experiment = self._get_experiment()

            # 초기 설정 로깅
            if self._config:
                self._experiment.config.update(self._config)

            # 코드 저장
            if self.save_code:
                self._save_code_snapshot()

        return self._experiment

    def _save_code_snapshot(self):
        """코드 스냅샷 저장"""
        try:
            # 현재 디렉토리의 Python 파일들 저장
            code_dir = Path(".")
            python_files = list(code_dir.rglob("*.py"))

            # 주요 파일들만 저장 (용량 제한)
            important_files = []
            for file_path in python_files:
                # 제외할 디렉토리들
                exclude_dirs = {'.git', '__pycache__', '.pytest_cache', 'outputs', 'wandb'}
                if not any(exclude_dir in file_path.parts for exclude_dir in exclude_dirs):
                    important_files.append(str(file_path))

            # Wandb에 코드 저장
            if important_files:
                wandb.save(important_files[:20])  # 최대 20개 파일
                logger.info(f"{len(important_files)} 개의 Python 파일을 Wandb에 저장했습니다.")

        except Exception as e:
            logger.warning(f"코드 스냅샷 저장 실패: {e}")

    def log_model_artifact(
            self,
            model_path: str,
            name: str,
            version: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
    ):
        """모델 아티팩트 로깅"""
        if not self.save_artifacts:
            return

        try:
            # 아티팩트 생성
            artifact = wandb.Artifact(
                name=name,
                type="model",
                description=f"Model checkpoint at {model_path}",
                metadata=metadata or {}
            )

            # 모델 파일 추가
            artifact.add_file(model_path)

            # 아티팩트 로깅
            self.experiment.log_artifact(artifact)

            logger.info(f"모델 아티팩트 저장 완료: {name}")

        except Exception as e:
            logger.error(f"모델 아티팩트 저장 실패: {e}")

    def log_dataset_artifact(
            self,
            dataset_path: str,
            name: str,
            description: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
    ):
        """데이터셋 아티팩트 로깅"""
        if not self.save_artifacts:
            return

        try:
            # 아티팩트 생성
            artifact = wandb.Artifact(
                name=name,
                type="dataset",
                description=description or f"Dataset at {dataset_path}",
                metadata=metadata or {}
            )

            # 데이터셋 추가 (디렉토리 또는 파일)
            if Path(dataset_path).is_dir():
                artifact.add_dir(dataset_path)
            else:
                artifact.add_file(dataset_path)

            # 아티팩트 로깅
            self.experiment.log_artifact(artifact)

            logger.info(f"데이터셋 아티팩트 저장 완료: {name}")

        except Exception as e:
            logger.error(f"데이터셋 아티팩트 저장 실패: {e}")


class WandbModelCheckpointCallback(Callback):
    """
    모델 체크포인트를 Wandb 아티팩트로 저장하는 콜백
    """

    def __init__(
            self,
            monitor: str = "val_loss",
            mode: str = "min",
            save_top_k: int = 3,
            save_last: bool = True
    ):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.best_k_models = {}
        self.kth_best_model_path = ""
        self.best_model_score = None
        self.best_model_path = ""

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """검증 종료 시 체크포인트 저장"""
        if not isinstance(trainer.logger, (WandbLogger, EnhancedWandbLogger)):
            return

        # 현재 메트릭 값 가져오기
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return

        current_score = current_score.item()

        # 최고 성능 체크
        if self._is_better_score(current_score):
            self.best_model_score = current_score

            # 체크포인트 저장
            checkpoint_path = f"checkpoints/best_model_epoch_{trainer.current_epoch}.ckpt"
            trainer.save_checkpoint(checkpoint_path)
            self.best_model_path = checkpoint_path

            # Wandb 아티팩트로 저장
            if hasattr(trainer.logger, 'log_model_artifact'):
                trainer.logger.log_model_artifact(
                    model_path=checkpoint_path,
                    name=f"best_model",
                    metadata={
                        "epoch": trainer.current_epoch,
                        "score": current_score,
                        "monitor": self.monitor,
                        "mode": self.mode
                    }
                )

            logger.info(f"새로운 최고 모델 저장: {self.monitor}={current_score:.4f}")

    def _is_better_score(self, score: float) -> bool:
        """점수가 더 좋은지 확인"""
        if self.best_model_score is None:
            return True

        if self.mode == "min":
            return score < self.best_model_score
        else:
            return score > self.best_model_score


class WandbExperimentManager:
    """
    Wandb 실험 관리 클래스
    """

    def __init__(self, project: str, entity: Optional[str] = None):
        self.project = project
        self.entity = entity

        # Wandb 로그인 확인
        self._ensure_wandb_login()

    def _ensure_wandb_login(self):
        """Wandb 로그인 확인"""
        try:
            wandb.login()
            logger.info("Wandb 로그인 확인 완료")
        except Exception as e:
            logger.error(f"Wandb 로그인 실패: {e}")
            raise

    def create_experiment(
            self,
            name: str,
            config: Dict[str, Any],
            tags: Optional[List[str]] = None,
            notes: Optional[str] = None
    ) -> EnhancedWandbLogger:
        """새로운 실험 생성"""
        # 실험 이름에 타임스탬프 추가
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        full_name = f"{name}_{timestamp}"

        # 로거 생성
        logger_instance = EnhancedWandbLogger(
            project=self.project,
            entity=self.entity,
            name=full_name,
            tags=tags,
            notes=notes,
            config=config
        )

        logger.info(f"새로운 실험 생성: {full_name}")
        return logger_instance

    def get_sweep_config(
            self,
            method: str = "bayes",
            metric_name: str = "val_loss",
            metric_goal: str = "minimize",
            parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Sweep 설정 반환"""
        if parameters is None:
            # 기본 하이퍼파라미터 범위
            parameters = {
                "learning_rate": {
                    "distribution": "log_uniform_values",
                    "min": 1e-6,
                    "max": 1e-3
                },
                "batch_size": {
                    "values": [8, 16, 32, 64]
                },
                "warmup_steps": {
                    "values": [100, 500, 1000]
                },
                "weight_decay": {
                    "distribution": "uniform",
                    "min": 0.0,
                    "max": 0.1
                }
            }

        sweep_config = {
            "method": method,
            "metric": {
                "name": metric_name,
                "goal": metric_goal
            },
            "parameters": parameters
        }

        return sweep_config

    def create_sweep(
            self,
            sweep_config: Dict[str, Any],
            project: Optional[str] = None
    ) -> str:
        """Sweep 생성"""
        sweep_id = wandb.sweep(
            sweep_config,
            project=project or self.project,
            entity=self.entity
        )

        logger.info(f"Sweep 생성 완료: {sweep_id}")
        return sweep_id

    def run_sweep(
            self,
            sweep_id: str,
            train_function: callable,
            count: Optional[int] = None
    ):
        """Sweep 실행"""
        wandb.agent(
            sweep_id,
            function=train_function,
            count=count,
            project=self.project,
            entity=self.entity
        )


# 사용 예시
if __name__ == "__main__":
    # 실험 관리자 생성
    experiment_manager = WandbExperimentManager(
        project="llm-modeling",
        entity="your-wandb-entity"  # 실제 entity로 변경
    )

    # 실험 설정
    config = {
        "model_name": "gpt2",
        "learning_rate": 2e-5,
        "batch_size": 16,
        "max_epochs": 3,
        "use_peft": True,
        "peft_config": {
            "r": 8,
            "lora_alpha": 16
        }
    }

    # 실험 생성
    logger_instance = experiment_manager.create_experiment(
        name="gpt2_classification",
        config=config,
        tags=["gpt2", "classification", "peft"],
        notes="GPT-2 모델을 사용한 텍스트 분류 실험"
    )

    print(f"실험 생성 완료: {logger_instance.experiment.name}")
    print(f"실험 URL: {logger_instance.experiment.url}")

    # Sweep 설정 예시
    sweep_config = experiment_manager.get_sweep_config(
        method="bayes",
        metric_name="val_accuracy",
        metric_goal="maximize",
        parameters={
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-6,
                "max": 1e-3
            },
            "peft_r": {
                "values": [4, 8, 16, 32]
            },
            "peft_alpha": {
                "values": [8, 16, 32, 64]
            }
        }
    )

    print("Sweep 설정:")
    print(json.dumps(sweep_config, indent=2))
```

## 🎛️ 고급 전이학습 전략

### 점진적 해동(Progressive Unfreezing)

**점진적 해동**은 처음에는 백본을 고정하고 분류기만 학습한 후, 단계적으로 백본의 레이어를 해동하는 기법이다.

```python
class ProgressiveUnfreezingModule(ImageTransferLearningModule):
    def __init__(self, *args, **kwargs):
        # 점진적 해동 관련 파라미터 추가
        self.unfreeze_epochs = kwargs.pop('unfreeze_epochs', [5, 10, 15])
        self.unfreeze_lr_factors = kwargs.pop('unfreeze_lr_factors', [0.1, 0.01, 0.001])

        super().__init__(*args, **kwargs)

        # 백본의 레이어 그룹 정의
        self.setup_layer_groups()

    def setup_layer_groups(self):
        """백본을 여러 그룹으로 나누기"""
        backbone_children = list(self.backbone.children())
        num_groups = 3

        # 레이어를 그룹으로 나누기
        group_size = len(backbone_children) // num_groups
        self.layer_groups = []

        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = (i + 1) * group_size if i < num_groups - 1 else len(backbone_children)
            group = backbone_children[start_idx:end_idx]
            self.layer_groups.append(group)

        print(f"백본을 {num_groups}개 그룹으로 분할했습니다.")
        for i, group in enumerate(self.layer_groups):
            print(f"그룹 {i + 1}: {len(group)}개 레이어")

    def unfreeze_layer_group(self, group_idx):
        """특정 레이어 그룹 해동"""
        if group_idx < len(self.layer_groups):
            for layer in self.layer_groups[group_idx]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"🔓 레이어 그룹 {group_idx + 1} 해동 완료!")

    def on_train_epoch_start(self):
        """에포크 시작 시 점진적 해동 체크"""
        current_epoch = self.current_epoch

        # 특정 에포크에서 레이어 그룹 해동
        for i, unfreeze_epoch in enumerate(self.unfreeze_epochs):
            if current_epoch == unfreeze_epoch and i < len(self.layer_groups):
                self.unfreeze_layer_group(i)

                # 학습률 조정
                if hasattr(self.trainer, 'optimizers'):
                    for optimizer in self.trainer.optimizers:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= self.unfreeze_lr_factors[i]

                print(f"학습률을 {self.unfreeze_lr_factors[i]}배로 조정했습니다.")


# 점진적 해동 모델 생성
progressive_model = ProgressiveUnfreezingModule(
    model_name='resnet50',
    num_classes=100,
    learning_rate=1e-3,
    freeze_backbone=True,
    unfreeze_epochs=[3, 6, 9],  # 3, 6, 9 에포크에서 해동
    unfreeze_lr_factors=[0.1, 0.1, 0.1]  # 각각 10%로 학습률 감소
)

print("점진적 해동 모델 생성 완료!")
# 출력: 점진적 해동 모델 생성 완료!
```

### 차별적 학습률(Differential Learning Rates)

**차별적 학습률**은 네트워크의 다른 부분에 서로 다른 학습률을 적용하는 기법이다. 일반적으로 사전 훈련된 부분은 낮은 학습률을, 새로 추가된 부분은 높은 학습률을 사용한다.

```python
class DifferentialLRModule(ImageTransferLearningModule):
    def __init__(self, *args, **kwargs):
        # 차별적 학습률 파라미터
        self.backbone_lr = kwargs.pop('backbone_lr', 1e-4)
        self.classifier_lr = kwargs.pop('classifier_lr', 1e-3)

        super().__init__(*args, freeze_backbone=False, **kwargs)

    def configure_optimizers(self):
        """차별적 학습률 설정"""
        # 파라미터 그룹 분리
        backbone_params = []
        classifier_params = []

        # 백본 파라미터
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                backbone_params.append(param)

        # 분류기 파라미터
        for name, param in self.classifier.named_parameters():
            if param.requires_grad:
                classifier_params.append(param)

        # 파라미터 그룹별 설정
        param_groups = [
            {
                'params': backbone_params,
                'lr': self.backbone_lr,
                'weight_decay': 1e-4
            },
            {
                'params': classifier_params,
                'lr': self.classifier_lr,
                'weight_decay': 1e-3
            }
        ]

        optimizer = AdamW(param_groups)

        # 스케줄러 (전체 학습률에 영향)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=30,
            eta_min=1e-6
        )

        print(f"차별적 학습률 설정:")
        print(f"  백본: {self.backbone_lr}")
        print(f"  분류기: {self.classifier_lr}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }


# 차별적 학습률 모델 생성
differential_lr_model = DifferentialLRModule(
    model_name='resnet50',
    num_classes=100,
    backbone_lr=1e-5,  # 백본은 매우 낮은 학습률
    classifier_lr=1e-3  # 분류기는 높은 학습률
)

print("차별적 학습률 모델 생성 완료!")
# 출력: 차별적 학습률 모델 생성 완료!
```

## 🤖 자연어 처리를 위한 전이학습

### BERT 기반 텍스트 분류

```python
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torchmetrics import Accuracy, F1Score


class BERTTransferLearningModule(pl.LightningModule):
    def __init__(
            self,
            model_name='bert-base-uncased',
            num_classes=2,
            learning_rate=2e-5,
            max_length=512,
            freeze_bert=False,
            dropout_rate=0.3
    ):
        super().__init__()
        self.save_hyperparameters()

        # BERT 모델과 설정 로드
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # BERT 고정 여부
        if freeze_bert:
            self.freeze_bert_layers()

        # 분류 헤드
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.config.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

        # 손실 함수와 메트릭
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.f1 = F1Score(task='multiclass', num_classes=num_classes)

    def freeze_bert_layers(self):
        """BERT 레이어 고정"""
        for param in self.bert.parameters():
            param.requires_grad = False
        print("🔒 BERT 레이어가 고정되었습니다.")

    def unfreeze_bert_layers(self, num_layers=None):
        """BERT 레이어 해동 (상위 N개 레이어만)"""
        if num_layers is None:
            # 전체 해동
            for param in self.bert.parameters():
                param.requires_grad = True
            print("🔓 모든 BERT 레이어가 해동되었습니다.")
        else:
            # 상위 N개 레이어만 해동
            total_layers = len(self.bert.encoder.layer)
            layers_to_unfreeze = total_layers - num_layers

            for i in range(layers_to_unfreeze, total_layers):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = True

            print(f"🔓 상위 {num_layers}개 BERT 레이어가 해동되었습니다.")

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """순전파"""
        # BERT 인코딩
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # [CLS] 토큰의 hidden state 사용
        pooled_output = outputs.pooler_output

        # 분류
        logits = self.classifier(pooled_output)
        return logits

    def training_step(self, batch, batch_idx):
        """학습 스텝"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)

        # 메트릭 계산
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, labels)

        # 로깅
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """검증 스텝"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, labels)
        self.f1(preds, labels)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.f1, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """옵티마이저 설정"""
        # AdamW 옵티마이저 (BERT에 최적화)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.01,
            eps=1e-8
        )

        # 선형 학습률 스케줄러 (워밍업 포함)
        num_training_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(0.1 * num_training_steps)

        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=num_warmup_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }


# NLP 데이터 모듈
class TextClassificationDataModule(pl.LightningDataModule):
    def __init__(
            self,
            tokenizer,
            train_texts,
            train_labels,
            val_texts,
            val_labels,
            max_length=512,
            batch_size=16
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_texts = train_texts
        self.train_labels = train_labels
        self.val_texts = val_texts
        self.val_labels = val_labels
        self.max_length = max_length
        self.batch_size = batch_size

    def setup(self, stage=None):
        """데이터셋 설정"""
        from torch.utils.data import Dataset

        class TextDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                text = str(self.texts[idx])
                label = self.labels[idx]

                # 토큰화
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )

                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(label, dtype=torch.long)
                }

        self.train_dataset = TextDataset(
            self.train_texts, self.train_labels,
            self.tokenizer, self.max_length
        )

        self.val_dataset = TextDataset(
            self.val_texts, self.val_labels,
            self.tokenizer, self.max_length
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )


# BERT 전이학습 모델 생성
bert_model = BERTTransferLearningModule(
    model_name='bert-base-uncased',
    num_classes=2,
    learning_rate=2e-5,
    freeze_bert=False
)

print("BERT 전이학습 모델 생성 완료!")
print(f"총 파라미터: {sum(p.numel() for p in bert_model.parameters()):,}")
print(f"학습 가능한 파라미터: {sum(p.numel() for p in bert_model.parameters() if p.requires_grad):,}")
# 출력: BERT 전이학습 모델 생성 완료!
# 출력: 총 파라미터: 109,483,778
# 출력: 학습 가능한 파라미터: 109,483,778
```

## 📊 전이학습 성능 모니터링과 콜백

### 전이학습 전용 콜백

```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np


class TransferLearningCallback(Callback):
    def __init__(self, unfreeze_epoch=5):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        self.frozen_performance = None
        self.unfrozen_performance = None

    def on_train_epoch_end(self, trainer, pl_module):
        """에포크 종료 시 메트릭 저장"""
        # 현재 메트릭 수집
        train_loss = trainer.callback_metrics.get('train_loss_epoch', 0)
        val_loss = trainer.callback_metrics.get('val_loss', 0)
        train_acc = trainer.callback_metrics.get('train_acc', 0)
        val_acc = trainer.callback_metrics.get('val_acc', 0)

        # 현재 학습률
        current_lr = trainer.optimizers[0].param_groups[0]['lr']

        # 메트릭 저장
        self.metrics_history['train_loss'].append(float(train_loss))
        self.metrics_history['val_loss'].append(float(val_loss))
        self.metrics_history['train_acc'].append(float(train_acc))
        self.metrics_history['val_acc'].append(float(val_acc))
        self.metrics_history['learning_rates'].append(current_lr)

        # 자동 해동 (모델에 unfreeze_backbone 메서드가 있는 경우)
        if (trainer.current_epoch == self.unfreeze_epoch and
                hasattr(pl_module, 'unfreeze_backbone')):

            # 고정 상태에서의 성능 저장
            self.frozen_performance = {
                'val_loss': float(val_loss),
                'val_acc': float(val_acc)
            }

            # 백본 해동
            pl_module.unfreeze_backbone()

            # 학습률 조정 (보통 더 낮게)
            for optimizer in trainer.optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1  # 10분의 1로 감소

            print(f"🔄 에포크 {self.unfreeze_epoch}에서 백본 해동 및 학습률 조정!")

    def on_train_end(self, trainer, pl_module):
        """학습 종료 시 성능 분석"""
        if len(self.metrics_history['val_acc']) > 0:
            self.unfrozen_performance = {
                'val_loss': self.metrics_history['val_loss'][-1],
                'val_acc': self.metrics_history['val_acc'][-1]
            }

            # 성능 비교 리포트
            self.generate_performance_report()

            # 학습 곡선 시각화
            self.plot_training_curves()

    def generate_performance_report(self):
        """성능 분석 리포트 생성"""
        print("\n" + "=" * 50)
        print("📊 전이학습 성능 분석 리포트")
        print("=" * 50)

        if self.frozen_performance:
            print(f"🔒 백본 고정 상태 (에포크 {self.unfreeze_epoch}):")
            print(f"   검증 손실: {self.frozen_performance['val_loss']:.4f}")
            print(f"   검증 정확도: {self.frozen_performance['val_acc']:.4f}")

        if self.unfrozen_performance:
            print(f"🔓 백본 해동 후 최종:")
            print(f"   검증 손실: {self.unfrozen_performance['val_loss']:.4f}")
            print(f"   검증 정확도: {self.unfrozen_performance['val_acc']:.4f}")

            if self.frozen_performance:
                acc_improvement = (self.unfrozen_performance['val_acc'] -
                                   self.frozen_performance['val_acc'])
                print(f"📈 정확도 개선: {acc_improvement:+.4f}")

        # 최고 성능
        best_val_acc = max(self.metrics_history['val_acc'])
        best_epoch = self.metrics_history['val_acc'].index(best_val_acc)
        print(f"🏆 최고 검증 정확도: {best_val_acc:.4f} (에포크 {best_epoch + 1})")

        print("=" * 50)

    def plot_training_curves(self):
        """학습 곡선 시각화"""
        epochs = range(1, len(self.metrics_history['train_loss']) + 1)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 손실 곡선
        ax1.plot(epochs, self.metrics_history['train_loss'], 'b-', label='학습 손실')
        ax1.plot(epochs, self.metrics_history['val_loss'], 'r-', label='검증 손실')
        if self.unfreeze_epoch < len(epochs):
            ax1.axvline(x=self.unfreeze_epoch, color='green', linestyle='--',
                        label='백본 해동')
        ax1.set_title('손실 변화')
        ax1.set_xlabel('에포크')
        ax1.set_ylabel('손실')
        ax1.legend()
        ax1.grid(True)

        # 정확도 곡선
        ax2.plot(epochs, self.metrics_history['train_acc'], 'b-', label='학습 정확도')
        ax2.plot(epochs, self.metrics_history['val_acc'], 'r-', label='검증 정확도')
        if self.unfreeze_epoch < len(epochs):
            ax2.axvline(x=self.unfreeze_epoch, color='green', linestyle='--',
                        label='백본 해동')
        ax2.set_title('정확도 변화')
        ax2.set_xlabel('에포크')
        ax2.set_ylabel('정확도')
        ax2.legend()
        ax2.grid(True)

        # 학습률 변화
        ax3.plot(epochs, self.metrics_history['learning_rates'], 'g-')
        ax3.set_title('학습률 변화')
        ax3.set_xlabel('에포크')
        ax3.set_ylabel('학습률')
        ax3.set_yscale('log')
        ax3.grid(True)

        # 과적합 분석
        train_val_gap = np.array(self.metrics_history['train_acc']) - np.array(self.metrics_history['val_acc'])
        ax4.plot(epochs, train_val_gap, 'purple', label='과적합 정도')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title('과적합 분석 (학습-검증 정확도 차이)')
        ax4.set_xlabel('에포크')
        ax4.set_ylabel('정확도 차이')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig('transfer_learning_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("📈 학습 곡선이 'transfer_learning_analysis.png'로 저장되었습니다!")


# 전이학습 콜백 사용
transfer_callback = TransferLearningCallback(unfreeze_epoch=5)

print("전이학습 전용 콜백 생성 완료!")
# 출력: 전이학습 전용 콜백 생성 완료!
```

## 🚀 완전한 실험 스크립트

이제 모든 구성요소를 조합하여 **완전한 LLM 학습 스크립트**를 작성해보자.

```python
# src/experiments/train_generator.py
"""텍스트 생성 모델 학습 스크립트"""

import os
import sys
from pathlib import Path
import argparse
import json
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import logging

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent))

# 프로젝트 모듈들
from models.generation import TextGenerationModule
from data.base_datamodule import TextGenerationDataModule
from callbacks.model_monitoring import (
    LLMModelMonitoringCallback,
    ModelSizeCallback,
    TextGenerationCallback,
    WandbModelCheckpointCallback
)
from utils.wandb_integration import EnhancedWandbLogger, WandbExperimentManager

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description="LLM 텍스트 생성 모델 학습")

    # 모델 설정
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="Hugging Face 모델 이름")
    parser.add_argument("--max_length", type=int, default=512,
                        help="최대 시퀀스 길이")

    # 생성 설정
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="최대 생성 토큰 수")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="생성 온도")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k 샘플링")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus) 샘플링")
    parser.add_argument("--do_sample", action="store_true", default=True,
                        help="샘플링 생성 여부")

    # 데이터 설정
    parser.add_argument("--dataset_name", type=str, default="wikitext",
                        help="Hugging Face 데이터셋 이름")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1",
                        help="데이터셋 설정")
    parser.add_argument("--text_column", type=str, default="text",
                        help="텍스트 컬럼 이름")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="배치 크기")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="최대 샘플 수 (디버깅용)")

    # 훈련 설정
    parser.add_argument("--max_epochs", type=int, default=3,
                        help="최대 에포크 수")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="학습률")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="가중치 감쇠")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="워밍업 스텝 수")

    # PEFT 설정
    parser.add_argument("--use_peft", action="store_true",
                        help="PEFT (LoRA) 사용 여부")
    parser.add_argument("--peft_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--peft_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--peft_dropout", type=float, default=0.1,
                        help="LoRA dropout")

    # 샘플 프롬프트 설정
    parser.add_argument("--sample_prompts", type=str, nargs="+",
                        default=[
                            "The future of artificial intelligence is",
                            "In a world where technology",
                            "Once upon a time in a distant galaxy"
                        ],
                        help="샘플 생성용 프롬프트")
    parser.add_argument("--generate_every_n_epochs", type=int, default=1,
                        help="몇 에포크마다 샘플 생성할지")

    # 하드웨어 설정
    parser.add_argument("--accelerator", type=str, default="auto",
                        help="가속기 종류")
    parser.add_argument("--devices", type=int, default=1,
                        help="사용할 디바이스 수")
    parser.add_argument("--precision", type=str, default="bf16-mixed",
                        help="정밀도")
    parser.add_argument("--strategy", type=str, default="auto",
                        help="분산 전략")

    # Wandb 설정
    parser.add_argument("--wandb_project", type=str, default="llm-generation",
                        help="Wandb 프로젝트 이름")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Wandb 엔티티")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="실험 이름")
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=[],
                        help="Wandb 태그 리스트")
    parser.add_argument("--offline", action="store_true",
                        help="오프라인 모드")

    # 기타 설정
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="출력 디렉토리")
    parser.add_argument("--seed", type=int, default=42,
                        help="랜덤 시드")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="데이터 로더 워커 수")
    parser.add_argument("--fast_dev_run", action="store_true",
                        help="빠른 개발 테스트 실행")

    return parser.parse_args()


def create_model(args):
    """모델 생성"""
    # PEFT 설정
    peft_config = None
    if args.use_peft:
        peft_config = {
            "r": args.peft_r,
            "lora_alpha": args.peft_alpha,
            "lora_dropout": args.peft_dropout,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        }

    # 모델 생성
    model = TextGenerationModule(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_epochs=args.max_epochs,
        use_peft=args.use_peft,
        peft_config=peft_config,
        model_max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample
    )

    return model


def create_data_module(args):
    """데이터 모듈 생성"""
    data_module = TextGenerationDataModule(
        tokenizer_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        text_column=args.text_column,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        max_samples=args.max_samples
    )

    return data_module


def setup_callbacks(args, wandb_logger):
    """콜백 설정"""
    callbacks = []

    # 모델 체크포인트
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.output_dir}/checkpoints",
        filename="best-{epoch:02d}-{val_perplexity:.2f}",
        monitor="val_perplexity",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)

    # 조기 종료
    early_stopping = EarlyStopping(
        monitor="val_perplexity",
        patience=3,
        mode="min",
        verbose=True,
        min_delta=0.1
    )
    callbacks.append(early_stopping)

    # 학습률 모니터링
    lr_monitor = LearningRateMonitor(
        logging_interval="step",
        log_momentum=True
    )
    callbacks.append(lr_monitor)

    # 커스텀 콜백들
    model_monitoring = LLMModelMonitoringCallback(
        log_every_n_steps=100,
        monitor_gradients=True,
        monitor_weights=True,
        monitor_resources=True
    )
    callbacks.append(model_monitoring)

    model_size_callback = ModelSizeCallback()
    callbacks.append(model_size_callback)

    # 텍스트 생성 콜백
    generation_callback = TextGenerationCallback(
        sample_prompts=args.sample_prompts,
        generate_every_n_epochs=args.generate_every_n_epochs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    callbacks.append(generation_callback)

    # Wandb 모델 체크포인트
    wandb_checkpoint = WandbModelCheckpointCallback(
        monitor="val_perplexity",
        mode="min",
        save_top_k=3
    )
    callbacks.append(wandb_checkpoint)

    return callbacks


def main():
    """메인 실행 함수"""
    # 인자 파싱
    args = parse_args()

    # 출력 디렉토리 생성
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    Path(f"{args.output_dir}/checkpoints").mkdir(exist_ok=True, parents=True)

    # 시드 설정
    pl.seed_everything(args.seed, workers=True)

    logger.info("=" * 60)
    logger.info("LLM 텍스트 생성 모델 학습 시작")
    logger.info("=" * 60)

    try:
        # Wandb 로거 설정 (분류 스크립트와 유사한 로직)
        if not args.offline:
            experiment_manager = WandbExperimentManager(
                project=args.wandb_project,
                entity=args.wandb_entity
            )

            config = vars(args)
            experiment_name = args.experiment_name or f"{args.model_name.replace('/', '-')}_generation"
            tags = args.wandb_tags + [args.model_name.split("/")[-1], "generation"]
            if args.use_peft:
                tags.append("peft")

            wandb_logger = experiment_manager.create_experiment(
                name=experiment_name,
                config=config,
                tags=tags
            )
            logger.info(f"Wandb 실험 URL: {wandb_logger.experiment.url}")
        else:
            from pytorch_lightning.loggers import CSVLogger
            wandb_logger = CSVLogger(args.output_dir, name="offline_logs")

        # 콜백 설정
        callbacks = setup_callbacks(args, wandb_logger)

        # 데이터 모듈 생성
        logger.info("데이터 모듈 생성 중...")
        data_module = create_data_module(args)
        data_module.prepare_data()
        data_module.setup("fit")

        # 모델 생성
        logger.info("모델 생성 중...")
        model = create_model(args)

        # 트레이너 설정
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator=args.accelerator,
            devices=args.devices,
            precision=args.precision,
            strategy=args.strategy,
            logger=wandb_logger,
            callbacks=callbacks,
            gradient_clip_val=1.0,
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            val_check_interval=1.0,
            log_every_n_steps=100,
            fast_dev_run=args.fast_dev_run,
            deterministic=True,
            benchmark=True
        )

        # 학습 시작
        logger.info("🚀 학습 시작!")
        trainer.fit(model, data_module)

        # 샘플 생성 테스트
        logger.info("📝 최종 샘플 생성:")
        for i, prompt in enumerate(args.sample_prompts[:3]):
            generated = model.generate_text(
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature
            )
            logger.info(f"프롬프트 {i + 1}: {prompt}")
            logger.info(f"생성 결과: {generated[0]}")
            logger.info("-" * 50)

        # 결과 저장
        best_model_path = trainer.checkpoint_callback.best_model_path
        results = {
            "experiment_name": wandb_logger.experiment.name if hasattr(wandb_logger, 'experiment') else "offline",
            "best_model_path": str(best_model_path),
            "config": vars(args)
        }

        results_path = f"{args.output_dir}/generation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info("=" * 60)
        logger.info("학습 완료!")
        logger.info(f"결과가 {results_path}에 저장되었습니다.")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"실험 중 오류 발생: {e}")
        raise

    finally:
        if not args.offline:
            try:
                import wandb
                wandb.finish()
            except:
                pass


if __name__ == "__main__":
    main()
```

## 💡 실무 활용 팁과 베스트 프랙티스

### 1. 메모리 효율적인 학습 전략

```python
# src/utils/memory_optimization.py
"""메모리 최적화를 위한 유틸리티"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy, FSDPStrategy
from typing import Dict, Any


class MemoryOptimizedTrainingStrategy:
    """메모리 최적화 전략 선택기"""

    @staticmethod
    def get_strategy_config(
            model_size: str,
            available_memory_gb: float,
            num_gpus: int
    ) -> Dict[str, Any]:
        """
        모델 크기와 사용 가능한 메모리에 따른 최적 전략 추천

        Args:
            model_size: "small", "medium", "large", "xl"
            available_memory_gb: GPU당 사용 가능 메모리 (GB)
            num_gpus: GPU 개수

        Returns:
            trainer 설정 딕셔너리
        """
        config = {
            "precision": "bf16-mixed",
            "accumulate_grad_batches": 1,
            "gradient_clip_val": 1.0
        }

        if model_size == "small":  # ~100M 파라미터
            if available_memory_gb >= 8:
                config.update({
                    "strategy": "auto",
                    "precision": "32"
                })
            else:
                config.update({
                    "strategy": "auto",
                    "precision": "bf16-mixed",
                    "accumulate_grad_batches": 2
                })

        elif model_size == "medium":  # ~300M 파라미터
            if available_memory_gb >= 16:
                config.update({
                    "strategy": "ddp" if num_gpus > 1 else "auto",
                    "precision": "bf16-mixed"
                })
            else:
                config.update({
                    "strategy": "ddp" if num_gpus > 1 else "auto",
                    "precision": "bf16-mixed",
                    "accumulate_grad_batches": 4
                })

        elif model_size == "large":  # ~1B 파라미터
            if available_memory_gb >= 24:
                config.update({
                    "strategy": FSDPStrategy() if num_gpus > 1 else "auto",
                    "precision": "bf16-mixed"
                })
            else:
                # DeepSpeed ZeRO Stage 2
                deepspeed_config = {
                    "zero_optimization": {
                        "stage": 2,
                        "offload_optimizer": {"device": "cpu"},
                        "allgather_partitions": True,
                        "allgather_bucket_size": 5e8,
                        "overlap_comm": True,
                        "reduce_scatter": True,
                        "reduce_bucket_size": 5e8,
                        "contiguous_gradients": True
                    },
                    "train_micro_batch_size_per_gpu": 1,
                    "gradient_accumulation_steps": 8
                }

                config.update({
                    "strategy": DeepSpeedStrategy(config=deepspeed_config),
                    "precision": "bf16-mixed",
                    "accumulate_grad_batches": 8
                })

        elif model_size == "xl":  # ~7B+ 파라미터
            # DeepSpeed ZeRO Stage 3 + CPU Offload
            deepspeed_config = {
                "zero_optimization": {
                    "stage": 3,
                    "offload_optimizer": {"device": "cpu"},
                    "offload_param": {"device": "cpu"},
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "sub_group_size": 1e9,
                    "reduce_bucket_size": "auto",
                    "stage3_prefetch_bucket_size": "auto",
                    "stage3_param_persistence_threshold": "auto",
                    "stage3_max_live_parameters": 1e9,
                    "stage3_max_reuse_distance": 1e9
                },
                "train_micro_batch_size_per_gpu": 1,
                "gradient_accumulation_steps": 16
            }

            config.update({
                "strategy": DeepSpeedStrategy(config=deepspeed_config),
                "precision": "bf16-mixed",
                "accumulate_grad_batches": 16
            })

        return config


# 사용 예시
def get_optimized_trainer_config(model_name: str, num_gpus: int) -> Dict[str, Any]:
    """모델에 따른 최적화된 trainer 설정 반환"""

    # 모델 크기 추정 (실제로는 더 정확한 방법 사용)
    model_size_map = {
        "gpt2": "small",
        "gpt2-medium": "medium",
        "gpt2-large": "large",
        "gpt2-xl": "large",
        "microsoft/DialoGPT-large": "large",
        "facebook/opt-1.3b": "large",
        "facebook/opt-6.7b": "xl",
        "meta-llama/Llama-2-7b": "xl"
    }

    model_size = model_size_map.get(model_name, "medium")

    # GPU 메모리 확인 (실제 환경에서는 동적으로 확인)
    if torch.cuda.is_available():
        available_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    else:
        available_memory_gb = 16  # 기본값

    strategy = MemoryOptimizedTrainingStrategy()
    return strategy.get_strategy_config(model_size, available_memory_gb, num_gpus)


print("메모리 최적화 전략 예시:")
config = get_optimized_trainer_config("gpt2-large", num_gpus=2)
for key, value in config.items():
    print(f"  {key}: {value}")
```

### 2. 실험 비교 및 분석

```python
# src/utils/experiment_analysis.py
"""실험 결과 분석 및 비교 도구"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
import numpy as np


class ExperimentAnalyzer:
    """Wandb 실험 결과 분석 클래스"""

    def __init__(self, project: str, entity: Optional[str] = None):
        self.project = project
        self.entity = entity
        self.api = wandb.Api()

    def get_experiment_data(self, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """실험 데이터 가져오기"""
        # 프로젝트에서 실행 목록 가져오기
        runs = self.api.runs(
            path=f"{self.entity}/{self.project}" if self.entity else self.project,
            filters=filters
        )

        # 데이터 수집
        data = []
        for run in runs:
            row = {
                "run_id": run.id,
                "name": run.name,
                "state": run.state,
                "created_at": run.created_at,
                "tags": run.tags,
                "notes": run.notes
            }

            # 설정 추가
            row.update({f"config_{k}": v for k, v in run.config.items()})

            # 요약 메트릭 추가
            row.update({f"summary_{k}": v for k, v in run.summary.items()})

            data.append(row)

        return pd.DataFrame(data)

    def compare_experiments(
            self,
            experiment_names: List[str],
            metrics: List[str] = ["val_loss", "val_accuracy"]
    ) -> None:
        """실험 비교 시각화"""
        # 데이터 가져오기
        filters = {"display_name": {"$in": experiment_names}}
        df = self.get_experiment_data(filters)

        if df.empty:
            print("비교할 실험 데이터가 없습니다.")
            return

        # 메트릭 비교 플롯
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 6 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            metric_col = f"summary_{metric}"
            if metric_col in df.columns:
                # 박스 플롯
                df_metric = df[df[metric_col].notna()]
                if not df_metric.empty:
                    sns.boxplot(data=df_metric, x="name", y=metric_col, ax=axes[i])
                    axes[i].set_title(f"{metric} 비교")
                    axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    def analyze_hyperparameter_impact(
            self,
            hyperparameter: str,
            target_metric: str = "val_accuracy"
    ) -> None:
        """하이퍼파라미터의 성능 영향 분석"""
        df = self.get_experiment_data()

        hp_col = f"config_{hyperparameter}"
        metric_col = f"summary_{target_metric}"

        if hp_col not in df.columns or metric_col not in df.columns:
            print(f"하이퍼파라미터 '{hyperparameter}' 또는 메트릭 '{target_metric}'을 찾을 수 없습니다.")
            return

        # 결측값 제거
        df_clean = df[[hp_col, metric_col]].dropna()

        if df_clean.empty:
            print("분석할 데이터가 없습니다.")
            return

        # 시각화
        plt.figure(figsize=(10, 6))

        # 수치형 하이퍼파라미터인 경우 산점도
        if pd.api.types.is_numeric_dtype(df_clean[hp_col]):
            plt.scatter(df_clean[hp_col], df_clean[metric_col], alpha=0.7)
            plt.xlabel(hyperparameter)
            plt.ylabel(target_metric)
            plt.title(f"{hyperparameter}의 {target_metric}에 대한 영향")

            # 추세선 추가
            z = np.polyfit(df_clean[hp_col], df_clean[metric_col], 1)
            p = np.poly1d(z)
            plt.plot(df_clean[hp_col], p(df_clean[hp_col]), "r--", alpha=0.8)

        else:
            # 범주형 하이퍼파라미터인 경우 박스 플롯
            sns.boxplot(data=df_clean, x=hp_col, y=metric_col)
            plt.xticks(rotation=45)
            plt.title(f"{hyperparameter}별 {target_metric} 분포")

        plt.tight_layout()
        plt.show()

        # 통계 요약
        if pd.api.types.is_numeric_dtype(df_clean[hp_col]):
            correlation = df_clean[hp_col].corr(df_clean[metric_col])
            print(f"상관계수: {correlation:.4f}")
        else:
            summary = df_clean.groupby(hp_col)[metric_col].agg(['mean', 'std', 'count'])
            print("그룹별 요약 통계:")
            print(summary)

    def find_best_experiments(
            self,
            metric: str = "val_accuracy",
            mode: str = "max",
            top_k: int = 5
    ) -> pd.DataFrame:
        """최고 성능 실험 찾기"""
        df = self.get_experiment_data()
        metric_col = f"summary_{metric}"

        if metric_col not in df.columns:
            print(f"메트릭 '{metric}'을 찾을 수 없습니다.")
            return pd.DataFrame()

        # 결측값 제거
        df_clean = df[df[metric_col].notna()]

        # 정렬
        ascending = (mode == "min")
        df_sorted = df_clean.sort_values(metric_col, ascending=ascending)

        # 상위 K개 선택
        top_experiments = df_sorted.head(top_k)

        # 주요 컬럼만 선택
        display_cols = ["name", "state", metric_col]
        config_cols = [col for col in df.columns if col.startswith("config_")]
        display_cols.extend(config_cols[:5])  # 주요 설정 5개만

        return top_experiments[display_cols]


# 사용 예시
if __name__ == "__main__":
    # 분석기 생성
    analyzer = ExperimentAnalyzer(
        project="llm-classification",
        entity="your-entity"
    )

    # 최고 성능 실험 찾기
    print("최고 성능 실험:")
    best_experiments = analyzer.find_best_experiments(
        metric="val_accuracy",
        mode="max",
        top_k=3
    )
    print(best_experiments)

    # 하이퍼파라미터 영향 분석
    analyzer.analyze_hyperparameter_impact(
        hyperparameter="learning_rate",
        target_metric="val_accuracy"
    )
```

### 3. 모델 배포를 위한 최적화

```python
# src/utils/model_optimization.py
"""모델 배포 최적화 도구"""

import torch
import pytorch_lightning as pl
from pathlib import Path
from typing import Optional, Dict, Any, Union
import onnx
import logging

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """모델 최적화 및 배포 준비 클래스"""

    @staticmethod
    def convert_to_torchscript(
            model: pl.LightningModule,
            save_path: str,
            example_input: Optional[torch.Tensor] = None
    ) -> str:
        """TorchScript로 변환"""
        model.eval()

        if example_input is None:
            # 기본 입력 생성 (BERT 스타일)
            batch_size, seq_len = 1, 128
            example_input = {
                "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
                "attention_mask": torch.ones(batch_size, seq_len)
            }

        # TorchScript 변환
        try:
            if isinstance(example_input, dict):
                # 딕셔너리 입력의 경우 trace 대신 script 사용
                scripted_model = torch.jit.script(model)
            else:
                scripted_model = torch.jit.trace(model, example_input)

            # 저장
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            scripted_model.save(save_path)

            logger.info(f"TorchScript 모델이 {save_path}에 저장되었습니다.")
            return save_path

        except Exception as e:
            logger.error(f"TorchScript 변환 실패: {e}")
            raise

    @staticmethod
    def convert_to_onnx(
            model: pl.LightningModule,
            save_path: str,
            example_input: Optional[Dict[str, torch.Tensor]] = None,
            dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    ) -> str:
        """ONNX로 변환"""
        model.eval()

        if example_input is None:
            batch_size, seq_len = 1, 128
            example_input = {
                "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
                "attention_mask": torch.ones(batch_size, seq_len)
            }

        if dynamic_axes is None:
            dynamic_axes = {
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "output": {0: "batch_size"}
            }

        try:
            # ONNX 내보내기
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            torch.onnx.export(
                model,
                tuple(example_input.values()),
                save_path,
                input_names=list(example_input.keys()),
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                opset_version=14,
                do_constant_folding=True,
                export_params=True
            )

            # ONNX 모델 검증
            onnx_model = onnx.load(save_path)
            onnx.checker.check_model(onnx_model)

            logger.info(f"ONNX 모델이 {save_path}에 저장되었습니다.")
            return save_path

        except Exception as e:
            logger.error(f"ONNX 변환 실패: {e}")
            raise

    @staticmethod
    def quantize_model(
            model: pl.LightningModule,
            save_path: str,
            quantization_type: str = "dynamic"
    ) -> str:
        """모델 양자화"""
        model.eval()

        try:
            if quantization_type == "dynamic":
                # 동적 양자화
                quantized_model = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear},  # 양자화할 레이어 타입
                    dtype=torch.qint8
                )
            else:
                raise ValueError(f"지원하지 않는 양자화 타입: {quantization_type}")

            # 저장
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(quantized_model.state_dict(), save_path)

            logger.info(f"양자화된 모델이 {save_path}에 저장되었습니다.")
            return save_path

        except Exception as e:
            logger.error(f"모델 양자화 실패: {e}")
            raise

    @staticmethod
    def optimize_for_inference(
            model: pl.LightningModule,
            output_dir: str
    ) -> Dict[str, str]:
        """추론 최적화된 여러 포맷으로 변환"""
        model.eval()
        model.freeze()  # PEFT 모델의 경우 파라미터 고정

        output_paths = {}

        try:
            # TorchScript 변환
            torchscript_path = f"{output_dir}/model.pt"
            output_paths["torchscript"] = ModelOptimizer.convert_to_torchscript(
                model, torchscript_path
            )

            # ONNX 변환
            onnx_path = f"{output_dir}/model.onnx"
            output_paths["onnx"] = ModelOptimizer.convert_to_onnx(
                model, onnx_path
            )

            # 양자화 모델
            quantized_path = f"{output_dir}/model_quantized.pth"
            output_paths["quantized"] = ModelOptimizer.quantize_model(
                model, quantized_path
            )

            logger.info("모든 최적화 완료:")
            for format_name, path in output_paths.items():
                logger.info(f"  {format_name}: {path}")

            return output_paths

        except Exception as e:
            logger.error(f"추론 최적화 중 오류 발생: {e}")
            raise


# 배포용 추론 클래스
class OptimizedInference:
    """최적화된 모델을 사용한 추론 클래스"""

    def __init__(self, model_path: str, tokenizer_name: str):
        self.model_path = model_path
        self.tokenizer_name = tokenizer_name

        # 토크나이저 로드
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # 모델 로드
        self._load_model()

    def _load_model(self):
        """모델 로드"""
        if self.model_path.endswith('.pt'):
            # TorchScript 모델
            self.model = torch.jit.load(self.model_path)
        elif self.model_path.endswith('.onnx'):
            # ONNX 모델
            import onnxruntime as ort
            self.model = ort.InferenceSession(self.model_path)
        else:
            # 일반 PyTorch 모델
            self.model = torch.load(self.model_path)

        logger.info(f"모델 로드 완료: {self.model_path}")

    def predict(self, texts: Union[str, list], max_length: int = 512):
        """텍스트 예측"""
        if isinstance(texts, str):
            texts = [texts]

        # 토크나이징
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )

        # 추론
        with torch.no_grad():
            if hasattr(self.model, 'run'):  # ONNX
                # ONNX 추론
                input_dict = {
                    name: inputs[name].numpy()
                    for name in inputs.keys()
                }
                outputs = self.model.run(None, input_dict)
                logits = torch.from_numpy(outputs[0])
            else:
                # PyTorch 추론
                outputs = self.model(**inputs)
                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs

        # 결과 처리
        predictions = torch.argmax(logits, dim=-1)
        probabilities = torch.softmax(logits, dim=-1)

        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist()
        }


# 사용 예시
if __name__ == "__main__":
    # 최적화 예시 (실제 모델과 경로 사용)
    print("모델 최적화 도구 사용 예시:")
    print("1. TorchScript 변환")
    print("2. ONNX 변환")
    print("3. 모델 양자화")
    print("4. 통합 최적화")

    # 추론 예시
    print("\n추론 클래스 사용 예시:")
    print("inference = OptimizedInference('model.pt', 'bert-base-uncased')")
    print("results = inference.predict(['This is a test sentence.'])")
```
