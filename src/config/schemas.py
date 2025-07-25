from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore


@dataclass
class ModelConfig:
    _target_: str
    name: str
    use_peft: bool = False
    peft_config: dict[str, Any] | None = None


@dataclass
class OptimizerConfig:
    _target_: str
    learning_rate: float
    weight_decay: float


@dataclass
class SchedulerConfig:
    _target_: str


@dataclass
class CosineSchedulerConfig(SchedulerConfig):
    _target_: str = "torch.optim.lr_scheduler.CosineAnnealingLR"
    T_max: int = 100
    eta_min: float = 1e-7


@dataclass
class StepSchedulerConfig(SchedulerConfig):
    _target_: str = "torch.optim.lr_scheduler.StepLR"
    step_size: int = 30
    gamma: float = 0.1


@dataclass
class ExponentialSchedulerConfig(SchedulerConfig):
    _target_: str = "torch.optim.lr_scheduler.ExponentialLR"
    gamma: float = 0.95


@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    warmup_steps: int
    checkpoint_path: str | None = None


@dataclass
class DataConfig:
    batch_size: int
    num_workers: int = 4
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1


@dataclass
class Config:
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training: TrainingConfig
    data: DataConfig

    experiment_name: str
    seed: int = 4321


config_store = ConfigStore.instance()
config_store.store(name="config", node=Config)

config_store.store(group="model", name="distilbert", node=ModelConfig)
config_store.store(group="model", name="distilgpt2", node=ModelConfig)

config_store.store(group="optimizer", name="adamw", node=OptimizerConfig)
config_store.store(group="optimizer", name="adam", node=OptimizerConfig)

config_store.store(group="scheduler", name="cosine", node=CosineSchedulerConfig)
config_store.store(group="scheduler", name="step", node=StepSchedulerConfig)
config_store.store(group="scheduler", name="exponential", node=ExponentialSchedulerConfig)

config_store.store(group="training", name="default", node=TrainingConfig)

config_store.store(group="data", name="default", node=DataConfig)
