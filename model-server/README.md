---
title: Driver Profiler Model Server
emoji: 🏎️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Driver Profiler — Model Server

FastAPI backend serving the Driver Profiler predictors (sector times,
consistency scoring, driving-style classification, telemetry delta vs
field) via XGBoost / Random Forest / LSTM models.

This runs on HuggingFace Spaces (Docker SDK) instead of Vercel because
the full model set — including the LSTM's TensorFlow dependency — comfortably
exceeds Vercel's 250MB serverless function limit. HF Spaces has no such
constraint.

## Endpoints

- `GET /` — health check, lists available circuits
- `POST /api/predict` — full prediction payload (see `app.py` for the request schema)

## Deploying

1. Create a new Space on huggingface.co → SDK: **Docker**
2. Push this `model-server/` folder's contents to the Space repo
3. Space builds automatically from the `Dockerfile` and starts serving on port 7860
4. Copy the Space's public URL (`https://<user>-<space>.hf.space`) into the
   frontend's `vercel.json` rewrite destination

## Local testing

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 7860
# then: curl -X POST http://localhost:7860/api/predict -H "Content-Type: application/json" -d '{"circuit":"Monaco"}'
```
