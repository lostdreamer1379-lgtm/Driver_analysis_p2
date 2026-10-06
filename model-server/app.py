"""
DRIVER PROFILER - Model Server (FastAPI, deployed on HuggingFace Spaces)

Why not Vercel: this project's models (XGBoost x9 + Random Forest x9 +
an LSTM in .keras/SavedModel format) run well past Vercel's 250MB
serverless function limit once TensorFlow is in the dependency tree.
HF Spaces has no such limit (Docker-based, up to 10GB free), so the
model server lives here and the React frontend on Vercel calls it over
HTTP via a lightweight rewrite — see frontend/vercel.json.

Endpoints:
  GET  /                 health check + available circuits/drivers
  POST /api/predict      full driver-profile prediction
"""

import os
import json
import traceback

import numpy as np
import joblib
import xgboost as xgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import utils

try:
    from tensorflow.keras.models import load_model as load_keras_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, 'models')
PRODUCTION_DIR = os.path.join(MODELS_DIR, 'production')

app = FastAPI(title="Driver Profiler API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # the frontend is a separate origin (Vercel) by design
    allow_methods=["*"],
    allow_headers=["*"],
)

_state = {'loaded': False}


class PredictRequest(BaseModel):
    circuit: str = "Silverstone"
    compound: str = "MEDIUM"
    driver: str = "VER"
    tyre_life: float = 10
    avg_speed: float = 210
    max_speed: float = 310
    avg_throttle_pct: float = 65
    brake_pct: float = 12
    recent_avg_lap_s: float = 90
    recent_std_s: float = 0.3
    lap_history: list | None = None  # optional [[lap_time, s1, s2, s3, avg_speed, throttle, brake, tyre_life], ...] x5 for LSTM


def _try_load_regressor(path_json, path_pkl):
    if os.path.exists(path_json):
        m = xgb.XGBRegressor()
        m.load_model(path_json)
        return m, 'xgboost'
    if os.path.exists(path_pkl):
        return joblib.load(path_pkl), 'sklearn'
    return None, None


def _try_load_classifier(path_json, path_pkl):
    if os.path.exists(path_json):
        m = xgb.XGBClassifier()
        m.load_model(path_json)
        return m, 'xgboost'
    if os.path.exists(path_pkl):
        return joblib.load(path_pkl), 'sklearn'
    return None, None


def load_models():
    if _state['loaded']:
        return

    encoders_path = os.path.join(MODELS_DIR, 'label_encoders.pkl')
    _state['encoders'] = joblib.load(encoders_path) if os.path.exists(encoders_path) else None

    selected_path = os.path.join(PRODUCTION_DIR, 'SELECTED_MODELS.json')
    selected = {}
    if os.path.exists(selected_path):
        with open(selected_path, 'r') as f:
            selected = json.load(f)
    _state['model_sources'] = selected

    # Sector 1/2/3 regressors (sector1 may be an LSTM)
    for sector in ['sector1', 'sector2', 'sector3']:
        best_name = selected.get(sector, 'xgboost')
        if best_name == 'lstm' and TF_AVAILABLE:
            lstm_path = os.path.join(PRODUCTION_DIR, f'model_{sector}_BEST.keras')
            if os.path.exists(lstm_path):
                _state[f'{sector}_model'] = load_keras_model(lstm_path)
                _state[f'{sector}_kind'] = 'lstm'
                continue
        json_p = os.path.join(PRODUCTION_DIR, f'model_{sector}_BEST.json')
        pkl_p = os.path.join(PRODUCTION_DIR, f'model_{sector}_BEST.pkl')
        if not os.path.exists(json_p) and not os.path.exists(pkl_p):
            json_p = os.path.join(MODELS_DIR, f'xgb_{sector}.json')
        model, kind = _try_load_regressor(json_p, pkl_p)
        _state[f'{sector}_model'] = model
        _state[f'{sector}_kind'] = kind
        scaler_p = os.path.join(PRODUCTION_DIR, f'scaler_{sector}_BEST.pkl')
        _state[f'{sector}_scaler'] = joblib.load(scaler_p) if os.path.exists(scaler_p) else None

    # Consistency (may also be an LSTM)
    best_consistency = selected.get('consistency', 'xgboost')
    if best_consistency == 'lstm' and TF_AVAILABLE:
        lstm_path = os.path.join(PRODUCTION_DIR, 'model_consistency_BEST.keras')
        if os.path.exists(lstm_path):
            _state['consistency_model'] = load_keras_model(lstm_path)
            _state['consistency_kind'] = 'lstm'
    if 'consistency_model' not in _state:
        json_p = os.path.join(PRODUCTION_DIR, 'model_consistency_BEST.json')
        pkl_p = os.path.join(PRODUCTION_DIR, 'model_consistency_BEST.pkl')
        if not os.path.exists(json_p) and not os.path.exists(pkl_p):
            json_p = os.path.join(MODELS_DIR, 'xgb_consistency.json')
        model, kind = _try_load_regressor(json_p, pkl_p)
        _state['consistency_model'] = model
        _state['consistency_kind'] = kind
        scaler_p = os.path.join(PRODUCTION_DIR, 'scaler_consistency_BEST.pkl')
        _state['consistency_scaler'] = joblib.load(scaler_p) if os.path.exists(scaler_p) else None

    # Driving style (classifier)
    json_p = os.path.join(PRODUCTION_DIR, 'model_style_BEST.json')
    pkl_p = os.path.join(PRODUCTION_DIR, 'model_style_BEST.pkl')
    if not os.path.exists(json_p) and not os.path.exists(pkl_p):
        json_p = os.path.join(MODELS_DIR, 'xgb_style.json')
    model, kind = _try_load_classifier(json_p, pkl_p)
    _state['style_model'] = model
    _state['style_kind'] = kind
    scaler_p = os.path.join(PRODUCTION_DIR, 'scaler_style_BEST.pkl')
    _state['style_scaler'] = joblib.load(scaler_p) if os.path.exists(scaler_p) else None

    style_names_meta = utils.get_metadata().get('style_names', {})
    _state['style_names'] = {int(k): v for k, v in style_names_meta.items()}

    # Telemetry delta
    json_p = os.path.join(PRODUCTION_DIR, 'model_telemetry_BEST.json')
    pkl_p = os.path.join(PRODUCTION_DIR, 'model_telemetry_BEST.pkl')
    if not os.path.exists(json_p) and not os.path.exists(pkl_p):
        json_p = os.path.join(MODELS_DIR, 'xgb_telemetry.json')
    model, kind = _try_load_regressor(json_p, pkl_p)
    _state['telemetry_model'] = model
    _state['telemetry_kind'] = kind
    scaler_p = os.path.join(PRODUCTION_DIR, 'scaler_telemetry_BEST.pkl')
    _state['telemetry_scaler'] = joblib.load(scaler_p) if os.path.exists(scaler_p) else None

    # SHAP explainers (optional, tree models only)
    for key in ['sector1', 'sector2', 'sector3', 'consistency', 'style', 'telemetry']:
        p = os.path.join(MODELS_DIR, f'shap_explainer_{key}.pkl')
        _state[f'{key}_explainer'] = joblib.load(p) if os.path.exists(p) else None

    _state['loaded'] = True


def _predict_scalar(kind, model, scaler, X):
    X_arr = np.array(X).reshape(1, -1)
    X_in = scaler.transform(X_arr) if (kind == 'sklearn' and scaler is not None) else X_arr
    return float(model.predict(X_in)[0]), X_arr


def _shap_top_factors(explainer, X_raw, feature_names, top_n=5):
    if explainer is None:
        return []
    try:
        shap_values = explainer(np.array(X_raw).reshape(1, -1))
        vals = shap_values.values[0]
        pairs = sorted(zip(feature_names, vals), key=lambda p: abs(p[1]), reverse=True)[:top_n]
        return [{'feature': name, 'impact': round(float(v), 4)} for name, v in pairs]
    except Exception:
        return []


@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "driver-profiler-model-server",
        "circuits": utils.list_circuits(),
        "tensorflow_available": TF_AVAILABLE,
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    try:
        load_models()
        meta = utils.get_metadata()
        encoders = _state['encoders']

        payload = req.dict()

        # --- Sector times ---
        sector_results = {}
        for sector in ['sector1', 'sector2', 'sector3']:
            kind = _state[f'{sector}_kind']
            model = _state[f'{sector}_model']

            if kind == 'lstm' and req.lap_history and len(req.lap_history) == meta.get('seq_len', 5):
                X_seq = np.array([req.lap_history], dtype=np.float32)
                pred = float(model.predict(X_seq, verbose=0).flatten()[0])
                X_for_shap = None
            else:
                X_vec = utils.build_feature_vector(sector, payload, encoders, meta)
                scaler = _state.get(f'{sector}_scaler')
                effective_kind = 'sklearn' if kind != 'xgboost' else 'xgboost'
                pred, X_for_shap = _predict_scalar(effective_kind if kind != 'lstm' else 'xgboost',
                                                     model if kind != 'lstm' else _state.get(f'{sector}_model'),
                                                     scaler, X_vec)
                X_for_shap = X_vec

            factors = _shap_top_factors(_state.get(f'{sector}_explainer'), X_for_shap, meta['sector_features']) if X_for_shap else []
            sector_results[sector] = {'time_s': round(max(0.0, pred), 3), 'shap_factors': factors}

        total_lap = sum(v['time_s'] for v in sector_results.values())

        # --- Consistency ---
        cons_kind = _state['consistency_kind']
        cons_model = _state['consistency_model']
        if cons_kind == 'lstm' and req.lap_history and len(req.lap_history) == meta.get('seq_len', 5):
            X_seq = np.array([req.lap_history], dtype=np.float32)
            cons_pred = float(cons_model.predict(X_seq, verbose=0).flatten()[0])
            cons_factors = []
        else:
            X_cons = utils.build_feature_vector('consistency', payload, encoders, meta)
            cons_pred, _ = _predict_scalar('sklearn' if cons_kind != 'xgboost' else 'xgboost',
                                            cons_model, _state.get('consistency_scaler'), X_cons)
            cons_factors = _shap_top_factors(_state.get('consistency_explainer'), X_cons, meta['consistency_features'])
        cons_pred = max(0.0, cons_pred)
        consistency_rating = 'very consistent' if cons_pred < 0.15 else ('consistent' if cons_pred < 0.35 else 'variable')

        # --- Driving style ---
        X_style = utils.build_feature_vector('style', payload, encoders, meta)
        style_kind = _state['style_kind']
        style_scaler = _state.get('style_scaler')
        X_style_arr = np.array(X_style).reshape(1, -1)
        X_style_in = style_scaler.transform(X_style_arr) if (style_kind == 'sklearn' and style_scaler is not None) else X_style_arr
        style_pred = int(_state['style_model'].predict(X_style_in)[0])
        style_label = _state['style_names'].get(style_pred, f'cluster_{style_pred}')
        style_factors = _shap_top_factors(_state.get('style_explainer'), X_style, meta['style_features'])

        # --- Telemetry delta vs field ---
        X_tel = utils.build_feature_vector('telemetry', payload, encoders, meta)
        tel_kind = _state['telemetry_kind']
        tel_pred, _ = _predict_scalar('sklearn' if tel_kind != 'xgboost' else 'xgboost',
                                       _state['telemetry_model'], _state.get('telemetry_scaler'), X_tel)
        tel_factors = _shap_top_factors(_state.get('telemetry_explainer'), X_tel, meta['telemetry_features'])

        return {
            "success": True,
            "input": payload,
            "predictions": {
                "sectors": sector_results,
                "estimated_lap_time_s": round(total_lap, 3),
                "consistency": {
                    "std_dev_s": round(cons_pred, 4),
                    "rating": consistency_rating,
                    "shap_factors": cons_factors,
                },
                "driving_style": {
                    "label": style_label,
                    "cluster_id": style_pred,
                    "shap_factors": style_factors,
                },
                "telemetry_delta": {
                    "seconds_vs_field_avg": round(tel_pred, 3),
                    "faster_than_field": tel_pred > 0,
                    "shap_factors": tel_factors,
                },
            },
            "model_sources": _state['model_sources'],
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "trace": traceback.format_exc()[:2000],
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
