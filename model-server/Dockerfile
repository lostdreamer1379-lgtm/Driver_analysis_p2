# Driver Profiler model server — deploys to HuggingFace Spaces (Docker SDK).
# HF Spaces has no serverless size limit, unlike Vercel, which is exactly
# why the heavy models (XGBoost + Random Forest + TensorFlow/LSTM) live here
# instead of as a Vercel API route.

FROM python:3.11-slim

WORKDIR /app

# System deps for xgboost / scikit-learn wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces expects the app to listen on port 7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
