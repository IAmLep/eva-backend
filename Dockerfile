"""

Version 3 working
"""

FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose port (this is just documentation, the actual port comes from PORT env var)
EXPOSE 8080

# Run the application with proper port binding for Cloud Run
# Using JSON array format for proper signal handling
CMD ["bash", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]