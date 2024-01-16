FROM python:3.8

WORKDIR /app

ENV HOST 0.0.0.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]

EXPOSE 8080
