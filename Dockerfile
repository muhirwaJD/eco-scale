# Eco-Scale serving image — runs the FastAPI API or the Streamlit dashboard.
# Multi-stage: the builder installs deps into a venv; the runtime image copies
# only that venv, so pip, its caches, and build cruft never ship.

# ---------- stage 1: build dependencies ----------
FROM python:3.12-slim AS builder
WORKDIR /app

# Self-contained venv we can copy wholesale into the runtime image.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Lean SERVING deps only (CPU-only torch, no training extras) — see
# requirements-serving.txt. This is what keeps the image small.
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# ---------- stage 2: runtime ----------
FROM python:3.12-slim AS runtime
WORKDIR /app

# Bring over just the prebuilt venv from the builder stage.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Application code + the deployable champion (per-run models/logs are excluded
# via .dockerignore).
COPY . .

# Run as a non-root user (security best practice).
RUN useradd --create-home --uid 1000 ecoscale && chown -R ecoscale /app
USER ecoscale

# FastAPI (8000) and Streamlit (8501); docker-compose picks which to run.
EXPOSE 8000 8501

# Default: serve the API.
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
