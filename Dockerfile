# Dockerfile
# Containerizes the FirstBank Nigeria customer segmentation ETL pipeline
# (BAN6800 Module 3). Builds a reproducible environment so the pipeline runs
# identically on any teammate's machine or in CI, addressing the
# reproducibility and collaboration requirement in the assignment brief.

FROM python:3.11-slim

LABEL maintainer="FirstBank Customer Segmentation Project - BAN6800"
LABEL description="ETL pipeline: ingest -> clean -> transform -> integrate -> validate -> bias_check -> anonymize"

# System dependencies for pandas/numpy wheels and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pipeline source code
COPY src/ ./src/
COPY tests/ ./tests/

# Create data and log directories with correct structure (mounted as volumes in production)
RUN mkdir -p data/raw data/interim data/processed logs great_expectations

# Run as a non-root user (least-privilege principle from the Module 3 Data Governance Framework)
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Basic healthcheck: confirm the Python environment and key imports are intact
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
    CMD python -c "import pandas, great_expectations, fairlearn, prefect" || exit 1

# Default command runs the full orchestrated DAG end to end
CMD ["python", "src/pipeline/pipeline_dag.py"]
