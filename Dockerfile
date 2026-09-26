FROM python:3.13-slim

WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt requirements.lock.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy source
COPY explainforge ./explainforge
COPY tests ./tests
COPY examples ./examples

# Smoke: run the deterministic benchmark end-to-end
RUN python explainforge/examples/run_demo.py

CMD ["python", "explainforge/examples/run_demo.py"]
