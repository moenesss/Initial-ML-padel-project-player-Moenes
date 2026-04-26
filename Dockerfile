# ══════════════════════════════════════════════════════════════════════════════
# Dockerfile — Padel Analytics · Flask ML API
# ══════════════════════════════════════════════════════════════════════════════

# Base image — Python 3.12 slim (lightweight)
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (so Docker caches this layer — faster rebuilds)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install mlflow and joblib (in case they're not in requirements.txt yet)
RUN pip install --no-cache-dir mlflow joblib

# Copy all project files into the container
COPY api.py .
COPY players_clean.csv .

# Create folders that the app needs
RUN mkdir -p /app/models /app/mlruns

# Expose Flask port
EXPOSE 5000

# Expose MLflow UI port
EXPOSE 5001

# Start the Flask API
CMD ["python", "api.py"]
