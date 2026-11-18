# Base image: Python 3.12 slim per handbook specification
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy dependency definition first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies before copying application code
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and configuration
COPY app/ ./app/
COPY src/ ./src/
COPY params.yaml .
COPY models/ ./models/

# Expose service port 8000
EXPOSE 8000

# Start Uvicorn ASGI server binding to 0.0.0.0:8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
