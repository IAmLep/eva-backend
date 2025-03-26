# Use the official Python base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Create and switch to a non-root user
RUN useradd --create-home appuser
WORKDIR /home/appuser/app

# Copy requirements file
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Create directory for mounted volume
RUN mkdir -p /home/appuser/data && chown -R appuser:appuser /home/appuser/data

# Switch to non-root user
USER appuser

# Use JSON array format for CMD to handle OS signals properly
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
