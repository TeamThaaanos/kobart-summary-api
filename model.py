from transformers import AutoTokenizer, BartForConditionalGeneration
import torch

# KoBART 모델 로드 함수
def load_kobart_model(model_path: str = "./saved_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
    model.eval()
    return tokenizer, model
