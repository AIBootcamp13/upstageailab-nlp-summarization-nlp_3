<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# KoBART 계열들 다 비교해줘

KoBART 계열 모델들을 체계적으로 비교해드리겠습니다:

## 주요 KoBART 계열 모델들

### 1. **KoBART (v1) - 원조 모델**

- **개발**: SKT-AI[^1]
- **특징**: 40GB 이상의 한국어 텍스트로 학습한 기본 모델
- **구조**: Encoder-Decoder (6층 각각, 124M 파라미터)[^1]
- **성능**:
    - NSMC: 90.24% 정확도[^1]
    - KorSTS: 81.66% (spearman)[^1]
    - Question Pair: 94.34% 정확도[^1]
- **용도**: 요약, 분류 등 기본적인 NLP 태스크
- **한계**: 프롬프트 처리 불가, 고정된 입출력 패턴


### 2. **KoBART-base-v2 - 개선된 버전**

- **개발**: gogamza[^2]
- **특징**:
    - **채팅 데이터 추가 학습**으로 긴 시퀀스 처리 능력 향상[^3]
    - **일부 instruction-following 능력** 보유
    - Hugging Face Transformers 완전 지원[^2]
- **성능**: NSMC 90.1% 정확도[^3]
- **개선점**:
    - 더 자연스러운 대화형 텍스트 이해
    - 긴 문서 처리 성능 향상
    - 표준화된 사용법 (transformers 라이브러리)


### 3. **KoBART-chat 시리즈**

- **개발**: haven-jeon, hyunwoongko 등[^4]
- **특징**:
    - **채팅/대화 데이터 특화** 파인튜닝
    - 간단한 챗봇 구현 가능[^4]
    - 대화형 응답 생성에 최적화
- **용도**: 챗봇, 대화 시스템, 질의응답
- **성능**: 특정 대화 태스크에서 우수한 성능


### 4. **특화 파인튜닝 모델들**

#### KoBART-summary-v3

- **개발**: EbanLee[^5]
- **특징**:
    - **짧은 문장 최적화**: v2 대비 더 자연스러운 단문 요약[^5]
    - **다중 데이터셋 파인튜닝**: 문서요약, 도서요약, 보고서 생성[^5]
    - 뉴스 요약, 학술 논문 요약에 특화[^5]


#### KoBART-persona-extraction-v2

- **개발**: NLPBada[^6]
- **특징**: 일상 대화에서 페르소나 추출 전문
- **버전 차이**: v1은 존댓말, v2는 반말 처리[^7]


## 성능 비교 분석

### **전반적인 성능 순위**

앞선 연구에 따르면[^8]:

1. **KoBART** ≈ **KoBERT** > **KoGPT-2**
2. 양방향 문맥 파악 능력이 단방향보다 우수[^8]

### **작업별 성능 특성**

#### 요약 작업

- **KoBART-summary-v3**: 짧은 문장 요약에 최적화[^5]
- **KoBART-v2**: 긴 문서 처리에 유리[^3]
- **원본 KoBART**: 일반적인 요약 태스크에 적합[^1]


#### 대화/채팅

- **KoBART-chat**: 대화형 응답 생성에 특화[^4]
- **KoBART-v2**: 채팅 데이터 학습으로 대화 이해 향상[^3]


#### 속도 vs 정확도

한 연구에서 KoBERT와 비교했을 때[^9]:

- **KoBART**: 학습 시간 약 15분, GPU 메모리 1초 단위, 정확도 63%
- **KoBERT**: 학습 시간 약 30시간, GPU 메모리 50-80ms, 정확도 77%


## 선택 가이드

### **일반적인 요약 작업**

→ **KoBART-base-v2** 추천 (표준화된 사용, 안정적 성능)

### **짧은 요약/뉴스 요약**

→ **KoBART-summary-v3** 추천 (단문 최적화)

### **대화형 애플리케이션**

→ **KoBART-chat** 또는 **KoBART-v2** 추천

### **연구/실험 목적**

→ **원본 KoBART** (기준점 역할)

### **상용 서비스**

→ **KoBART-base-v2** (Hugging Face 완전 지원, 안정성)

## 결론

**KoBART-base-v2**가 현재 가장 **균형잡힌 선택**입니다. 원본의 장점을 유지하면서도 긴 시퀀스 처리, 표준화된 사용법, 일부 instruction-following 능력까지 갖추고 있어 대부분의 용도에 적합합니다[^3][^2].

특별한 용도(초단문 요약, 대화형 등)가 아니라면 KoBART-base-v2로 시작하여 필요에 따라 특화 모델로 전환하는 것이 효율적인 접근법입니다.

<div style="text-align: center">⁂</div>

[^1]: https://github.com/SKT-AI/KoBART

[^2]: https://huggingface.co/gogamza/kobart-base-v2

[^3]: https://www.promptlayer.com/models/kobart

[^4]: https://github.com/haven-jeon/KoBART-chatbot

[^5]: https://model.aibase.com/models/details/1915693260686909441

[^6]: https://huggingface.co/NLPBada/kobart-chat-persona-extraction-v2/blame/11bb3f02a08caa50a8c50b9ed62178c869cc1a60/README.md

[^7]: https://huggingface.co/NLPBada/kobart-chat-persona-extraction-v2/discussions/1

[^8]: http://journal.dcs.or.kr/_PR/view/?aidx=38618\&bidx=3485

[^9]: https://heroeswillnotdie.tistory.com/18

[^10]: https://koreascience.kr/article/CFKO202226455348918.pdf

[^11]: https://velog.io/@jx7789/KoBART를-활용한-카카오톡-대화-요약-서비스5ft.추가-개선

[^12]: https://ki-it.com/xml/35065/35065.pdf

[^13]: https://dataloop.ai/library/model/gogamza_kobart-base-v2/

[^14]: https://github.com/seujung/KoBART-summarization

[^15]: https://huggingface.co/gogamza/kobart-base-v2/discussions/1/files

[^16]: https://dacon.io/en/competitions/official/235829/codeshare/4047

[^17]: https://github.com/SKT-AI/KoBART/issues/18

[^18]: https://github.com/hyunwoongko/kobart-transformers

[^19]: https://www.kci.go.kr/kciportal/landing/article.kci?arti_id=ART003220006

[^20]: https://mongsangcole.tistory.com/6

