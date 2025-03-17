# summary.py
import re

stopwords = set([
  "네", "음", "아", "어", "그렇군요", "예", "아하",
  "고객님", "상담원", "저기요", "혹시","고객:","상담원:",
  "어떤 문제 있으신가요?", "어떻게 도와드릴까요?", "알려주시겠어요?","무엇을",
  "불편을 끼쳐드려 죄송합니다", "정말 죄송합니다", "양해 부탁드립니다", "이용에 불편을 드려 죄송합니다",
  "네, 알겠습니다", "네, 감사합니다", "추가로 궁금한 사항 있으시면 언제든 문의 주세요.", "좋은 하루 보내세요","안녕하세요.","안녕하세요",
  "빠르게 처리해드리겠습니다", "확인 후 연락드리겠습니다",
  "입니다", "있습니다", "되나요", "가능한가요", "도와드릴까요",
  "은", "는", "이", "가", "을", "를", "의", "에", "와", "과", "에서", "하다"
])

def remove_stopwords(text):
    if isinstance(text, str):
        for phrase in stopwords:
            if len(phrase) > 3:
                text = text.replace(phrase, "")
        words = text.split()
        filtered = [word for word in words if word not in stopwords]
        return " ".join(filtered)
    return text

def preprocess(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text