# 일상 대화 요약 대회

## Team

| <img src="https://avatars.githubusercontent.com/u/119947716?v=4" width="150" height="150" style="border-radius: 50%;"/> | <img src="https://avatars.githubusercontent.com/u/178347552?v=4" width="150" height="150" style="border-radius: 50%;"/> | <img src="https://avatars.githubusercontent.com/u/4915390?v=4" width="150" height="150" style="border-radius: 50%;"/> |
|:-----------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------:|
|                                          [조재형](https://github.com/Bitefriend)                                           |                                           [김두환](https://github.com/korea202a)                                           |                                          [조의영](https://github.com/yuiyeong)                                           |
|                                                        팀장, W.I.P                                                        |                                                          W.I.P                                                          |                                                         W.I.P                                                         |

## Overview

### Environment

- **OS**: Linux (Ubuntu/CentOS)
- **CPU**: AMD Ryzen Threadripper 3960X 24-Core Processor (24 cores / 48 threads)
- **RAM**: 256GB (251GiB available)
- **GPU**: NVIDIA GeForce RTX 3090 (24GB VRAM)
- **CUDA**: 12.2
- **Python**: >=3.11

### Requirements

- **Core Dependencies**
    - PyTorch 2.7.1+
    - Transformers 4.53.2+
    - Lightning 2.5.2+
    - Datasets 4.0.0+
- **Training Optimization**
    - PEFT (Parameter-Efficient Fine-Tuning) 0.16.0+
    - Accelerate 1.9.0+
    - DeepSpeed 0.17.2+
- **Data & Visualization**
    - NumPy 2.3.1+, Pandas 2.3.1+
    - Matplotlib 3.10.3+, Seaborn 0.13.2+
- **Experiment Management**
    - Weights & Biases (wandb) 0.21.0+
    - Hydra-core 1.3.2+

## Competition Info

### Overview

**Dialogue Summarization** 경진대회는 주어진 데이터를 활용하여 일상 대화에 대한 요약을 효과적으로 생성하는 모델을 개발하는 대회입니다.

일상생활에서 이루어지는 다양한 대화(회의, 토의, 일상 대화 등)를 입력으로 받아 핵심적인 내용을 요약하는 자연어 생성 모델을 구축하는 것이 목표입니다.

**평가 방식**

- ROUGE-1-F1, ROUGE-2-F1, ROUGE-L-F1의 평균 점수로 최종 성능 평가
- Multi-Reference Dataset으로 3개의 정답 요약문과 비교하여 평가
- 한국어 형태소 분석기를 통한 토큰화 후 ROUGE 점수 산출

### Timeline

- **2025년 7월 25일 (금) 10:00** - 대회 시작
- **2025년 8월 6일 (수) 19:00** - 최종 제출 마감
- **대회 기간**: 2주

### Dataset Info

- **Training**: 12,457개 대화-요약 쌍
- **Development**: 499개
- **Test**: 250개
- **Hidden Test**: 249개 (최종 평가용)

**데이터 구성**

- **대화문**: 최소 2명~최대 7명이 참여하는 대화 (최소 2턴~최대 60턴)
- **화자 구분**: `#Person"N"#:` 형식으로 발화자 표시
- **요약문**: 각 대화에 대응하는 요약문 (평가 시 3개의 다중 참조 요약문 사용)

## Data Description

### Dataset Overview

- **DialogSum 데이터셋 기반** (한국어 번역 버전)
- **라이선스**: CC BY-NC-SA 4.0 license
- **언어**: 한국어
- **도메인**: 일상 대화, 회의, 상담 등 다양한 상황의 대화

## Components

### Directory

```
.
├── README.md
├── configs                        # Hydra 설정 파일들
│   ├── config.yaml                # 메인 설정
│   ├── data                       # 데이터 관련 설정
│   ├── model                      # 모델별 설정
│   ├── optimizer                  # 옵티마이저 설정
│   ├── scheduler                  # 학습률 스케줄러 설정
│   └── training                   # 학습 관련 설정
├── data
│   ├── fonts                      # 폰트 데이터
│   └── raw                        # 원본 데이터섹
├── docs                           # 프로젝트 문서
├── notebooks                      # 실험 및 분석용 노트북
│   ├── notebook_template.ipynb    # Jupyter Notebook 템플릿
├── scripts                        # shell 스크립트
└── src                            # Source Code Root
    ├── config                     # 설정 및 Hydra 설정 Schema
    │   └── schemas.py
    ├── data                       # 데이터 처리 및 데이터 모듈
    ├── models                     # LLM 모델 구현
    ├── scripts                    # 실험 및 예측 스크립트
    ├── training                   # 실험 관련 pytorch lightning 유틸리티
    └── utils                      # 공통 유틸리티
```

## Getting Started

### Cloud Instance 환경 설정 with Shell Script

**Python 버전과 의존성 관리자(Poetry) 및 Python 관련 설치 및 설정**

1. GPU 서버에 SSH 로 로그인한 다음, 아래 명령어를 입력하여 환경 설정 스크립트 다운로드한다.
    ```bash
    wget https://gist.githubusercontent.com/yuiyeong/8ae3f167e97aeff90785a4ccda41e5fe/raw/d5e030ea64bbd9c41ce2b4c825bc03c86f0c3dac/setup_env.sh
    ```

2. 다운로드 받은 스크립트를 실행 파일로 변경한다.
    ```bash
    chmod +x setup_env.sh
    ```

3. 실행 파일을 실행한다.
    ```bash
    ./setup_env.sh
    ```

**이 스크립트는 다음 내용을 설정한다.**

- 시스템 업데이트 및 Python 빌드 의존성 설치
- /workspace 작업 디렉토리 생성
- Python 3.11 conda 환경 구성
- Poetry 설치 및 경로 설정
- 환경 변수 설정 (HOME, PATH, PYTHONPATH)
- SSH 로그인 시 자동 설정들

### 프로젝트 클론 및 의존성 설치

1. GPU 서버에 SSH 로 로그인한 다음, 아래 명령어로 이 Repository 를 Clone 한다.
    ```bash
    git clone https://github.com/AIBootcamp13/upstageailab-nlp-summarization-nlp_3.git
    ```

2. 다음 명령어를 실행해서 프로젝트의 환경 설정을 마친다.
    ```bash
    cd upstageailab-nlp-summarization-nlp_3.git # 이제 경로는 "/workspace/upstageailab-nlp-summarization-nlp_3" 이다.

    poetry install --with dev
    poetry run pre-commit install
    ```

### 데이터 준비

```bash
# 데이터 다운로드
wget [.gz.tar 확장자의 dataset url]

# 압축 해제
tar -xzf data.tar.gz -C data/raw/
```

### EDA

- W.I.P

### Data Processing

- W.I.P

## Modeling

### Model Description

- W.I.P

### Modeling Process

- W.I.P

## Result

### Leader Board

- W.I.P

### Presentation

- W.I.P

## etc

### Meeting Log

W.I.P - 회의록 링크 추가 예정

### Reference

- [DialogSum Dataset](https://github.com/cylnlp/dialogsum)
- [ROUGE Metric](https://en.wikipedia.org/wiki/ROUGE_(metric))
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
- [Hydra](https://hydra.cc/docs/intro/)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://huggingface.co/docs/peft/index)
