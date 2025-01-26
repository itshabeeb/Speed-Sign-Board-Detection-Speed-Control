FROM python:3.9-slim-buster
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cahche-dir -r requirements.txt
COPY . .
CMD ["python", "train.py"]
