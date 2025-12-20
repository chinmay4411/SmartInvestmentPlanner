FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5500

CMD ["gunicorn", "-b", "0.0.0.0:5500", "app:app"]
