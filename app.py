from flask import Flask, request, jsonify
from model import load_kobart_model
from summary import preprocess, remove_stopwords  # 전처리 함수
import torch

app = Flask(__name__)

# KoBART 모델 로드 (서버 시작 시)
tokenizer, model = load_kobart_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 요약 처리 함수
def process_summary(consult_id, text):
    try:
        # 1️⃣ 전처리 적용
        cleaned = preprocess(text)
        cleaned = remove_stopwords(cleaned)

        # 2️⃣ KoBART 요약 생성
        inputs = tokenizer(cleaned, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # token_type_ids 제거 (BART는 필요 없음)
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

        # 결과 반환 (URL 같은 거 저장은 여기선 제외)
        return {
            "summary_text": summary_text,
            "consult_id": consult_id
        }

    except Exception as e:
        return {"error": str(e)}

# 기본 라우트
@app.route("/")
def home():
    return "KoBART Flask 서버가 실행 중입니다."

# 요약 요청 처리 라우트
@app.route("/summary", methods=["POST"])
def summary():
    try:
        data = request.get_json()
        consult_id = data.get("consult_id", "")
        text = data.get("text", "")

        if not consult_id or not text:
            return jsonify({"error": "consult_id 또는 text가 누락되었습니다"}), 400

        result = process_summary(consult_id, text)

        if "error" in result:
            return jsonify(result), 500

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
