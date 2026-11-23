FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8080

COPY requirements.txt .

RUN pip install --no-cache-dir \
    --default-timeout=300 \
    --retries 10 \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.8.0

RUN pip install --no-cache-dir \
    --default-timeout=300 \
    --retries 10 \
    -r requirements.txt

COPY app.py .
COPY src ./src
COPY data/processed ./data/processed
COPY data/evaluation ./data/evaluation

EXPOSE 8080

CMD ["streamlit", "run", "app.py"]