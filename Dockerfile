# Use the official Python base image
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Use Cloud Run's PORT environment variable
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
