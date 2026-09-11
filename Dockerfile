FROM python:3.11-slim

# Set system environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8000

WORKDIR /app

# Install minimal build and runtime tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy fine-tuned model artifacts and application code
COPY models/intent_setfit /app/models/intent_setfit
COPY data/processed/apple_playbook_vault.json /app/data/processed/apple_playbook_vault.json
COPY src /app/src

# Healthcheck probe
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

EXPOSE 8000

CMD ["uvicorn", "src.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
