FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 기본 패키지
RUN apt-get update && apt-get install -y python3 python3-pip git && \
    ln -s /usr/bin/python3 /usr/bin/python

# 작업 폴더 지정
WORKDIR /app

# requirements.txt 설치
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 앱 코드 복사
COPY app.py .
COPY saved_model ./saved_model
COPY handler.py .
COPY model.py .
COPY summary.py .
# 포트 설정
EXPOSE 8000

# Flask 서버 실행
CMD ["python", "app.py"]
