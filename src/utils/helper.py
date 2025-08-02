import re
import pandas as pd

from datetime import datetime

import pytorch_lightning as pl

from src import config


DATETIME_FORMAT = "%Y%m%d%H%M%S"


def fix_random_seed(seed_num: int):
    """모든 랜덤 시드를 고정하여 실험의 재현성을 보장하는 함수"""
    pl.seed_everything(seed_num)


def add_timestamp_prefix(filename: str) -> str:
    timestamp = datetime.now(config.TIMEZONE).strftime(DATETIME_FORMAT)
    return f"{timestamp}_{filename}"


def handle_missing_values(text):
    
    # NULL 값이나 None 처리
    if pd.isna(text) or text is None:
        return ""
    
    if isinstance(text, (int, float)):
        return str(text)
    return str(text)

def fix_encoding(text):
    # UTF-8 인코딩 문제 해결
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    # HTML 엔티티 디코딩
    import html
    text = html.unescape(text)
    
    return text

def normalize_korean(text):
    # 유니코드 정규화 (NFD -> NFC)
    import unicodedata
    text = unicodedata.normalize('NFC', text)
    
    # 한글 자모 분리 문제 해결
    text = re.sub(r'[\u1100-\u11FF\u3130-\u318F]', '', text)
    
    return text

def context_aware_removal(text):
    """문맥을 고려한 태그 처리"""
    
    # HTML 태그 특징: 소문자, 알려진 태그명
    html_pattern = r'<(div|p|span|a|img|br|hr|h[1-6]|ul|ol|li|table|tr|td|strong|b|em|i)[^>]*>'
    text = re.sub(html_pattern, '', text, flags=re.IGNORECASE)
    
    # 닫는 태그
    text = re.sub(r'</(div|p|span|a|h[1-6]|ul|ol|li|table|tr|td|strong|b|em|i)>', '', text, flags=re.IGNORECASE)
    
    # 화자 태그는 그대로 두거나 변환
    # <이름> 패턴을 이름: 으로 변환
    text = re.sub(r'<([A-Za-z가-힣\s]+)>', r'\1:', text)
    
    return text


def remove_markup(text):
    # HTML 태그 제거
    #text = re.sub(r'<[^>]+>', '', text)
    text = context_aware_removal(text)
    
    # XML 태그 제거
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    
    # 마크다운 링크 제거
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # 마크다운 이미지 제거
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', text)
    
    return text

def handle_emoticons(text):
    # 이모티콘 정규화
    emoticon_patterns = {
        r':\)|:-\)|:\]|:-\]|\(:|\(-:': '[HAPPY]',
        r':\(|:-\(|:\[|:-\[|\):|\)-:': '[SAD]',
        r'ㅋ{2,}|크{2,}|킼{2,}': '[LAUGH]',
        r'ㅎ{2,}|하{2,}|흐{2,}': '[LAUGH]',
        r'ㅜ{2,}|우{2,}|으{2,}': '[CRY]'
    }
    
    for pattern, replacement in emoticon_patterns.items():
        text = re.sub(pattern, replacement, text)
    
    return text

def clean_special_chars(text):
    # URL 제거
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
    
    # 이메일 제거
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    
    # 전화번호 제거
    text = re.sub(r'\b\d{2,3}-\d{3,4}-\d{4}\b', '[PHONE]', text)
    
    # 해시태그 처리
    text = re.sub(r'#\w+', '[HASHTAG]', text)
    
    # 멘션 처리
    text = re.sub(r'@\w+', '[MENTION]', text)
    
    return text

def reduce_repetition(text):
    # 반복되는 문자 축약 (3개 이상 → 2개)
    #text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'([^\d])\1{2,}', r'\1\1', text)  # 숫자가 아닌 문자의 반복만 축약

    # 반복되는 문장부호 정리
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{3,}', '...', text)
    
    return text

def normalize_numbers(text):
    # 연도 정규화
    text = re.sub(r'\b(19|20)\d{2}년?\b', '[YEAR]', text)
    
    # 날짜 정규화
    text = re.sub(r'\b\d{1,2}월\s?\d{1,2}일?\b', '[DATE]', text)
    
    # 시간 정규화
    text = re.sub(r'\b\d{1,2}:\d{2}\b', '[TIME]', text)
    
    # 큰 숫자 정규화
    text = re.sub(r'\b\d+만\b', '[NUMBER]만', text)
    text = re.sub(r'\b\d+억\b', '[NUMBER]억', text)
    
    return text

def preprocess_dialogue(text):
    # 화자 표시 정규화
    text = re.sub(r'#Person\d+#:', '[SPEAKER]:', text)
    
    # 대화 구분자 정리
    text = re.sub(r'\n+', ' [SEP] ', text)
    
    # 감정 표현 보존
    text = re.sub(r'([ㅋㅎㅜㅠ])\1+', r'\1\1', text)
    
    return text

def basic_cleaning(text):
    # 불필요한 공백 정리
    text = re.sub(r'\s+', ' ', text)  # 연속 공백을 하나로
    text = text.strip()  # 앞뒤 공백 제거
    
    # 특수 공백 문자 정리
    text = re.sub(r'[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000\uFEFF]', ' ', text)
    
    return text

def preprocess(text):
    """종합적인 전처리 파이프라인"""
    
    # 1. 기본 검증
    text = handle_missing_values(text)
    if not text:
        return ""
    
    # 2. 인코딩 및 정규화
    text = fix_encoding(text)
    text = normalize_korean(text)
    
    # 3. 마크업 제거
    text = remove_markup(text)
    
    # 4. 특수 문자 처리
    #text = handle_emoticons(text)
    #text = clean_special_chars(text)
    
    # 5. 반복 문자 정리
    text = reduce_repetition(text)
    
    # 6. 숫자 정규화
    #text = normalize_numbers(text)
    
    # 7. 대화 형태 처리
    text = preprocess_dialogue(text)
    
    # 8. 최종 정리
    text = basic_cleaning(text)
    
    return text