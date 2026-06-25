# Eco-Scale full-stack image — builds the React console and serves it together
# with the FastAPI API behind ONE port. Used by docker-compose (port 8000, the
# command is overridden there) and by Hugging Face Spaces (port 7860).

# ---------- stage 1: build the React console ----------
FROM node:22-slim AS web
WORKDIR /web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
# Same-origin API: the FastAPI app serves this build, so calls go to /config etc.
RUN VITE_API_URL="" npm run build

# ---------- stage 2: python dependencies ----------
FROM python:3.12-slim AS builder
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# ---------- stage 3: runtime ----------
FROM python:3.12-slim AS runtime
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# App code + the deployable champion, plus the built console from stage 1.
COPY . .
COPY --from=web /web/dist ./web/dist

# Run as a non-root user (security best practice).
RUN useradd --create-home --uid 1000 ecoscale && chown -R ecoscale /app
USER ecoscale

# 7860 = Hugging Face Spaces default; 8000/8501 used by docker-compose locally.
EXPOSE 7860 8000 8501

# Default serves the console + API on 7860 (Hugging Face). docker-compose
# overrides this command to use port 8000 / run Streamlit.
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "7860"]
