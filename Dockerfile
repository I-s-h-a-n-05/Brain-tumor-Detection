FROM python:3.11-slim

# HuggingFace Spaces requires a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy and install dependencies first (layer caching)
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all app files
COPY --chown=user . /app

# HuggingFace Spaces requires port 7860
CMD ["gunicorn", "app:app", "--workers", "1", "--threads", "2", "--timeout", "120", "--bind", "0.0.0.0:7860"]