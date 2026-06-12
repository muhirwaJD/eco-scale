# Eco-Scale serving image — runs the FastAPI API or the Streamlit dashboard.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (better layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the code and the deployable champion (per-run models/logs are excluded
# via .dockerignore).
COPY . .

# FastAPI (8000) and Streamlit (8501); docker-compose picks which to run.
EXPOSE 8000 8501

# Default: serve the API.
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
