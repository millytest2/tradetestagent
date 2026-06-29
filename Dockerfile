FROM python:3.11-slim

WORKDIR /app

# Install system deps for cryptography / xgboost
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Init DB on startup, then run the live bot
CMD ["python", "main.py", "--live"]
