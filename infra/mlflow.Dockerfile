FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir mlflow==3.6.0

EXPOSE 5000

# Use PORT environment variable (defaults to 5000)
CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port ${PORT:-5000} --backend-store-uri file:///mlruns --default-artifact-root /mlruns"]
