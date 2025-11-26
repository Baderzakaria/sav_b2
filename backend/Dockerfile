FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if needed (e.g. for some python packages)
# build-essential for compiling some python libs
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy necessary files from root
COPY orchestrator.py /app/
COPY utils_runtime.py /app/
COPY clean_gpu.py /app/
COPY apps /app/apps
COPY monitoring /app/monitoring
COPY prompts /app/prompts
# Also need data directory structure, maybe inputs
# Assuming input csv is in data/cleaned/free_tweet_export-latest.csv
# We might need to mount data volume or copy if static
COPY data /app/data

# Copy backend code
COPY backend /app/backend

# Expose port
EXPOSE 8000

# Command to run the backend
# Use PORT environment variable (Railway sets this automatically)
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
