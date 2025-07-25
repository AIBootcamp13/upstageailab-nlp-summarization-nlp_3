from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint

from src import config


def get_callbacks(experiment_name: str, monitor: str = "val_loss") -> list[Callback]:
    checkpoint_dir = config.CHECKPOINT_DIR / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return [
        # 모델 체크포인트 저장
        ModelCheckpoint(
            dirpath=checkpoint_dir,  # 체크포인트 파일을 저장할 디렉토리 경로
            filename="periodic-{epoch:02d}-{val_loss:.3f}-{step}",  # 체크포인트 파일명 패턴
            every_n_train_steps=5000,  # 5000 스텝마다 저장
            save_top_k=-1,  # 모든 체크포인트 보존
            save_last=True,  # 마지막 에포크 모델 저장 여부
            verbose=True,  # 로그 출력 여부
        ),
        # 조기 종료
        EarlyStopping(
            monitor=monitor,  # 모니터링할 메트릭 이름
            min_delta=0.001,
            mode="min" if "loss" in monitor else "max",  # 메트릭 최적화 방향 ('min' 또는 'max')
            patience=10,  # 개선이 없어도 기다릴 에포크 수
            verbose=True,  # 로그 출력 여부
        ),
        # 학습률 모니터링
        LearningRateMonitor(logging_interval="step"),
    ]
