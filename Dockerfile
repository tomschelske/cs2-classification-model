# Slim serving image for the CS2 win-probability dashboard.
# Ships the model + the precomputed featured round + the frontend — no parser
# stack (demoparser2/awpy/libomp) and no LightGBM, because parsing is offline and
# the deployed model is linear. Result: a small, boring, reliable image.
FROM python:3.13-slim

WORKDIR /app

# Install deps first so this layer caches across code changes.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Only the serving code (api.py is self-contained — it does not import the
# parsing/training modules) plus the small artifacts it serves.
COPY cs2wp/__init__.py cs2wp/api.py ./cs2wp/
COPY models/model.pkl ./models/model.pkl
COPY data/round_navi_falcons.json ./data/round_navi_falcons.json
COPY frontend/ ./frontend/

# Cloud hosts inject $PORT; default to 8000 for local runs.
ENV PORT=8000
EXPOSE 8000

# Shell form so ${PORT} is expanded at runtime.
CMD ["sh", "-c", "uvicorn cs2wp.api:app --host 0.0.0.0 --port ${PORT}"]
