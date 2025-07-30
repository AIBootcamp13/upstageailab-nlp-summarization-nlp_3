
# ✅ 1. 기본 라이브러리 불러오기
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from transformers import AutoTokenizer

# ✅ 2. 한글 폰트 설정 (NanumBarunGothic.ttf 사용)

font_path = r"C:\Users\재형띠\Desktop\코딩친구들\upstageailab-nlp-summarization-nlp_3\data\fonts\NanumBarunGothic.ttf"
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'NanumBarunGothic'
plt.rcParams['axes.unicode_minus'] = False

# 폰트 경로: data/fonts/NanumBarunGothic.ttf
fontprop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = fontprop.get_name()
plt.rcParams['axes.unicode_minus'] = False  # 마이너스(-) 깨짐 방지

# ✅ 3. 데이터 로드 (src/data)
train_path = r"C:\Users\재형띠\Desktop\코딩친구들\upstageailab-nlp-summarization-nlp_3\src\data\train.csv"
test_path = r"C:\Users\재형띠\Desktop\코딩친구들\upstageailab-nlp-summarization-nlp_3\src\data\test.csv"

train = pd.read_csv(train_path, encoding="utf-8")  # or cp949 if needed
test = pd.read_csv(test_path, encoding="utf-8")

# ✅ 4. 기본 정보 확인
print("✅ train.csv shape:", train.shape)
print("✅ test.csv shape:", test.shape)
print("\n📌 train 컬럼 목록:", train.columns.tolist())
print("\n📌 train 샘플:\n", train.head())

# ✅ 5. 결측치 & 중복 확인
print("\n📌 결측치:\n", train.isnull().sum())
print("\n📌 중복 행 개수:", train.duplicated().sum())

# ✅ 6. 텍스트 길이 통계 분석
train['text_length'] = train['dialogue'].apply(len)
print("\n📌 text 길이 통계:\n", train['text_length'].describe())

print("📌 토픽별 샘플 텍스트 (최대 5개까지 출력):")
unique_topics = sorted(train["topic"].unique())[:5]  # 너무 많으면 제한

for topic in unique_topics:
    sample_text = train[train["topic"] == topic]["dialogue"].values[0][:300]
    print(f"\n🟡 [Topic: {topic}] 샘플 텍스트 (앞 300자):\n{sample_text}")

# ✅ 1. 텍스트 길이 분포 시각화
plt.figure(figsize=(10, 6))
sns.histplot(train['text_length'], bins=50, kde=True, color='royalblue')
plt.title("텍스트 길이 분포", fontsize=16)
plt.xlabel("문자 수", fontsize=13)
plt.ylabel("샘플 수", fontsize=13)
plt.tight_layout()
plt.grid(True)
plt.show()

# ✅ 2. 요약 텍스트 길이 분포 (summary)
train['summary_len'] = train['summary'].apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(train['summary_len'], bins=50, kde=True, color='salmon')
plt.title("요약 텍스트 길이 분포")
plt.xlabel("문자 수")
plt.ylabel("샘플 수")
plt.tight_layout()
plt.savefig("summary_length_distribution.png", dpi=200)
plt.close()

# ✅ 3. 레이블 분포 (상위 100개)
top_topics = train['topic'].value_counts().nlargest(100)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_topics.index, y=top_topics.values, palette="Set3")
plt.title("Topic 상위 100개 분포")
plt.xticks(rotation=75)
plt.tight_layout()
plt.savefig("top100_topic_distribution.png", dpi=200)
plt.close()

# ✅ 4. 고유 토픽 개수 출력
print("고유 Topic 개수:", train['topic'].nunique())

# ✅ 5. 특수문자/숫자 포함 비율
import re

def count_special_ratio(text):
    return len(re.findall(r"[^\w\s]", text)) / len(text)

train['special_ratio'] = train['dialogue'].apply(count_special_ratio)
print("평균 특수문자 비율:", train['special_ratio'].mean())

# ✅ 6. tokenizer 기준 토큰 길이
tokenizer = AutoTokenizer.from_pretrained("hyunwoongko/kobart") # 요약 task에 적합
train['dialogue_token_len'] = train['dialogue'].apply(lambda x: len(tokenizer.encode(x, truncation=False)))
train['summary_token_len'] = train['summary'].apply(lambda x: len(tokenizer.encode(x, truncation=False)))

plt.figure(figsize=(10, 5))
sns.histplot(train['dialogue_token_len'], color="dodgerblue", label="입력")
sns.histplot(train['summary_token_len'], color="tomato", label="요약")
plt.title("Token 길이 분포")
plt.xlabel("Token 수")
plt.legend()
plt.tight_layout()
plt.savefig("token_length_comparison.png", dpi=200)
plt.close()

# 요약 vs 입력 길이 비율 분포
# → 평균값이 너무 높거나 낮으면 모델이 잘 못 배움
#→ 보통 3~8 사이 분포가 안정적
train['dialogue_len'] = train['dialogue'].apply(len)
train['summary_len'] = train['summary'].apply(len)
train['length_ratio'] = train['dialogue_len'] / train['summary_len']

plt.figure(figsize=(10, 6))
sns.histplot(train['length_ratio'], bins=50, kde=True, color="purple")
plt.title("입력 길이 / 요약 길이 비율 분포", fontsize=16)
plt.xlabel("비율 (dialogue / summary)")
plt.ylabel("샘플 수")
plt.tight_layout()
plt.savefig("dialogue_summary_ratio.png", dpi=200)
plt.close()

# tokenizer 적용 후 입력/요약 토큰 수
# → max_seq_len, max_target_len 설정 근거가 됨
# → ex) 95% quantile 기준으로 자르는 게 좋음
tokenizer = AutoTokenizer.from_pretrained("hyunwoongko/kobart")

train['input_token_len'] = train['dialogue'].apply(lambda x: len(tokenizer.encode(x, truncation=False)))
train['output_token_len'] = train['summary'].apply(lambda x: len(tokenizer.encode(x, truncation=False)))

plt.figure(figsize=(10, 6))
sns.histplot(train['input_token_len'], color="dodgerblue", label="입력", kde=True)
sns.histplot(train['output_token_len'], color="tomato", label="요약", kde=True)
plt.title("Token 길이 분포 (tokenizer 적용 후)", fontsize=16)
plt.xlabel("토큰 수")
plt.legend()
plt.tight_layout()
plt.savefig("token_length_distribution.png", dpi=200)
plt.close()

# topic별 요약 길이 차이 확인
plt.figure(figsize=(14, 6))
sns.boxplot(x='topic', y='summary_len', data=train[train['topic'].isin(train['topic'].value_counts().nlargest(10).index)])
plt.title("상위 10개 Topic별 요약 길이 분포", fontsize=16)
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig("summary_length_per_topic.png", dpi=200)
plt.close()

# 중복 대화 탐지
duplicate_dialogues = train['dialogue'].duplicated().sum()
print(f"📌 중복된 dialogue 수: {duplicate_dialogues}개 ({duplicate_dialogues/train.shape[0]*100:.2f}%)")