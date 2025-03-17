import re
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

# ✅ 전처리 함수
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

def preprocess(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def remove_stopwords(text):
    if isinstance(text, str):
        for phrase in stopwords:
            if len(phrase) > 3:
                text = text.replace(phrase, "")
        words = text.split()
        filtered = [word for word in words if word not in stopwords]
        return " ".join(filtered)
    return text

# ✅ 모델 로드
MODEL_PATH = "./saved_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = BartForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ RunPod 실행 함수
def handler(event):
    try:
        input_text = event["input"]["text"]
        consult_id = event["input"].get("consult_id", "")

        # 전처리
        cleaned = preprocess(input_text)
        cleaned = remove_stopwords(cleaned)

        # 요약 생성
        inputs = tokenizer(cleaned, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs.pop("token_type_ids", None)

        summary_ids = model.generate(
            **inputs,
            max_length=450,
            min_length=200,
            num_beams=5,
            length_penalty=1.2,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return {
            "summary": summary_text,
            "consult_id": consult_id
        }

    except Exception as e:
        return { "error": str(e) }
