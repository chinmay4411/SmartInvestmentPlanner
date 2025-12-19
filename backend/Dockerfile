
# 1️⃣ Use stable Python (THIS fixes sklearn error)
FROM python:3.10-slim

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Copy only requirements first (faster builds)
COPY requirements.txt .

# 4️⃣ Install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 5️⃣ Copy backend code
COPY . .

# 6️⃣ Expose Flask port
EXPOSE 5500

# 7️⃣ Start Flask using gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5500"]
