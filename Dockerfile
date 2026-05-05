FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY api.py .
COPY CNNModel.h5 .

# Hugging Face Spaces uses port 7860
ENV PORT=7860
EXPOSE 7860

# Use uvicorn for FastAPI
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
