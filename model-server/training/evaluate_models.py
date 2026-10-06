"""
DRIVER PROFILER - Model Evaluation
Scores every trained candidate per predictor (including LSTM where it was
trained), generates SHAP plots for the tree-based models, and promotes the
single best model per predictor into ../models/production/.

Note on SHAP + LSTM: SHAP's DeepExplainer is heavy and version-fragile, and
the sequence LSTM isn't the kind of thing stakeholders read SHAP bars for
anyway (lap-history patterns, not single-point feature attribution) — so
SHAP here is generated only for the XGBoost candidates, same as Strategy
Master. The LSTM is still fully evaluated and can still win promotion.
"""

import os
import json
import numpy as np
import joblib
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
)
import xgboost as xgb
import shap

try:
    from tensorflow.keras.models import load_model as load_keras_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, '../models')
PRODUCTION_DIR = os.path.join(MODELS_DIR, 'production')
REPORT_DIR = os.path.join(HERE, '../../evaluation_report')
METADATA_PATH = os.path.join(MODELS_DIR, 'model_metadata.json')

os.makedirs(PRODUCTION_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def load_metadata():
    with open(METADATA_PATH, 'r') as f:
        return json.load(f)


def evaluate_regression(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"      {name:<18} MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def evaluate_classification(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"      {name:<18} Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


def _shap_plot(xgb_model, X_test, feature_names, out_name, explainer_name):
    try:
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer(X_test[:200])
        joblib.dump(explainer, os.path.join(MODELS_DIR, explainer_name))
        plt.figure()
        shap.summary_plot(shap_values, X_test[:200], feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, out_name), dpi=120)
        plt.close()
    except Exception as e:
        print(f"      SHAP skipped: {str(e)[:60]}")


def evaluate_sector(meta):
    print("\n[1/4] Sector breakdown (regression x3) — lower MAE is better")
    results = {}
    for sector in ['sector1', 'sector2', 'sector3']:
        print(f"   -- {sector} --")
        split = joblib.load(os.path.join(MODELS_DIR, f'test_split_{sector}.pkl'))
        X_test, X_test_s, y_test = split['X_test'], split['X_test_s'], split['y_test']

        sector_results = {}
        lr = joblib.load(os.path.join(MODELS_DIR, f'linear_{sector}.pkl'))
        sector_results['linear_regression'] = evaluate_regression('Linear Regression', y_test, lr.predict(X_test_s))

        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(os.path.join(MODELS_DIR, f'xgb_{sector}.json'))
        sector_results['xgboost'] = evaluate_regression('XGBoost', y_test, xgb_model.predict(X_test))

        rf = joblib.load(os.path.join(MODELS_DIR, f'rf_{sector}.pkl'))
        sector_results['random_forest'] = evaluate_regression('Random Forest', y_test, rf.predict(X_test))

        lstm_path = os.path.join(MODELS_DIR, f'lstm_{sector}.keras')
        if sector == 'sector1' and TF_AVAILABLE and os.path.exists(lstm_path):
            lstm_split_path = os.path.join(MODELS_DIR, 'test_split_sector1_lstm.pkl')
            if os.path.exists(lstm_split_path):
                lstm_split = joblib.load(lstm_split_path)
                lstm = load_keras_model(lstm_path)
                preds = lstm.predict(lstm_split['X_test'], verbose=0).flatten()
                sector_results['lstm'] = evaluate_regression('LSTM', lstm_split['y_test'], preds)

        best_name = min(sector_results, key=lambda k: sector_results[k]['mae'])
        print(f"      -> Best: {best_name} (MAE={sector_results[best_name]['mae']:.4f})")

        _shap_plot(xgb_model, X_test, meta['sector_features'], f'shap_{sector}.png', f'shap_explainer_{sector}.pkl')
        _promote(best_name, sector)
        results[sector] = {'scores': sector_results, 'best': best_name}

    return results


def evaluate_consistency(meta):
    print("\n[2/4] Consistency scorer (regression) — lower MAE is better")
    split = joblib.load(os.path.join(MODELS_DIR, 'test_split_consistency.pkl'))
    X_test, X_test_s, y_test = split['X_test'], split['X_test_s'], split['y_test']

    results = {}
    lr = joblib.load(os.path.join(MODELS_DIR, 'linear_consistency.pkl'))
    results['linear_regression'] = evaluate_regression('Linear Regression', y_test, lr.predict(X_test_s))

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(os.path.join(MODELS_DIR, 'xgb_consistency.json'))
    results['xgboost'] = evaluate_regression('XGBoost', y_test, xgb_model.predict(X_test))

    rf = joblib.load(os.path.join(MODELS_DIR, 'rf_consistency.pkl'))
    results['random_forest'] = evaluate_regression('Random Forest', y_test, rf.predict(X_test))

    lstm_path = os.path.join(MODELS_DIR, 'lstm_consistency.keras')
    if TF_AVAILABLE and os.path.exists(lstm_path):
        lstm_split_path = os.path.join(MODELS_DIR, 'test_split_consistency_lstm.pkl')
        if os.path.exists(lstm_split_path):
            lstm_split = joblib.load(lstm_split_path)
            lstm = load_keras_model(lstm_path)
            preds = lstm.predict(lstm_split['X_test'], verbose=0).flatten()
            results['lstm'] = evaluate_regression('LSTM', lstm_split['y_test'], preds)

    best_name = min(results, key=lambda k: results[k]['mae'])
    print(f"   -> Best: {best_name} (MAE={results[best_name]['mae']:.4f})")

    _shap_plot(xgb_model, X_test, meta['consistency_features'], 'shap_consistency.png', 'shap_explainer_consistency.pkl')
    _promote(best_name, 'consistency')
    return results, best_name


def evaluate_driving_style(meta):
    print("\n[3/4] Driving style (classification) — higher F1 is better")
    split = joblib.load(os.path.join(MODELS_DIR, 'test_split_style.pkl'))
    X_test, X_test_s, y_test = split['X_test'], split['X_test_s'], split['y_test']

    results = {}
    logr = joblib.load(os.path.join(MODELS_DIR, 'logr_style.pkl'))
    results['logistic_regression'] = evaluate_classification('Logistic Reg', y_test, logr.predict(X_test_s))

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(MODELS_DIR, 'xgb_style.json'))
    results['xgboost'] = evaluate_classification('XGBoost', y_test, xgb_model.predict(X_test))

    rf = joblib.load(os.path.join(MODELS_DIR, 'rf_style.pkl'))
    results['random_forest'] = evaluate_classification('Random Forest', y_test, rf.predict(X_test))

    best_name = max(results, key=lambda k: results[k]['f1'])
    print(f"   -> Best: {best_name} (F1={results[best_name]['f1']:.4f})")

    _shap_plot(xgb_model, X_test, meta['style_features'], 'shap_style.png', 'shap_explainer_style.pkl')
    _promote(best_name, 'style')
    return results, best_name


def evaluate_telemetry(meta):
    print("\n[4/4] Telemetry delta vs field (regression) — lower MAE is better")
    split = joblib.load(os.path.join(MODELS_DIR, 'test_split_telemetry.pkl'))
    X_test, X_test_s, y_test = split['X_test'], split['X_test_s'], split['y_test']

    results = {}
    lr = joblib.load(os.path.join(MODELS_DIR, 'linear_telemetry.pkl'))
    results['linear_regression'] = evaluate_regression('Linear Regression', y_test, lr.predict(X_test_s))

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(os.path.join(MODELS_DIR, 'xgb_telemetry.json'))
    results['xgboost'] = evaluate_regression('XGBoost', y_test, xgb_model.predict(X_test))

    rf = joblib.load(os.path.join(MODELS_DIR, 'rf_telemetry.pkl'))
    results['random_forest'] = evaluate_regression('Random Forest', y_test, rf.predict(X_test))

    best_name = min(results, key=lambda k: results[k]['mae'])
    print(f"   -> Best: {best_name} (MAE={results[best_name]['mae']:.4f})")

    _shap_plot(xgb_model, X_test, meta['telemetry_features'], 'shap_telemetry.png', 'shap_explainer_telemetry.pkl')
    _promote(best_name, 'telemetry')
    return results, best_name


MODEL_FILE_MAP = {
    'sector1': {
        'linear_regression': ('linear_sector1.pkl', 'scaler_sector1.pkl'),
        'xgboost': ('xgb_sector1.json', None),
        'random_forest': ('rf_sector1.pkl', None),
        'lstm': ('lstm_sector1.keras', None),
    },
    'sector2': {
        'linear_regression': ('linear_sector2.pkl', 'scaler_sector2.pkl'),
        'xgboost': ('xgb_sector2.json', None),
        'random_forest': ('rf_sector2.pkl', None),
    },
    'sector3': {
        'linear_regression': ('linear_sector3.pkl', 'scaler_sector3.pkl'),
        'xgboost': ('xgb_sector3.json', None),
        'random_forest': ('rf_sector3.pkl', None),
    },
    'consistency': {
        'linear_regression': ('linear_consistency.pkl', 'scaler_consistency.pkl'),
        'xgboost': ('xgb_consistency.json', None),
        'random_forest': ('rf_consistency.pkl', None),
        'lstm': ('lstm_consistency.keras', None),
    },
    'style': {
        'logistic_regression': ('logr_style.pkl', 'scaler_style.pkl'),
        'xgboost': ('xgb_style.json', None),
        'random_forest': ('rf_style.pkl', None),
    },
    'telemetry': {
        'linear_regression': ('linear_telemetry.pkl', 'scaler_telemetry.pkl'),
        'xgboost': ('xgb_telemetry.json', None),
        'random_forest': ('rf_telemetry.pkl', None),
    },
}


def _promote(best_name, predictor):
    model_file, scaler_file = MODEL_FILE_MAP[predictor][best_name]
    src = os.path.join(MODELS_DIR, model_file)
    if not os.path.exists(src):
        print(f"      WARNING: {src} not found, cannot promote {predictor}")
        return
    ext = os.path.splitext(model_file)[1]
    dst = os.path.join(PRODUCTION_DIR, f'model_{predictor}_BEST{ext}')
    shutil.copy2(src, dst)
    if scaler_file:
        shutil.copy2(
            os.path.join(MODELS_DIR, scaler_file),
            os.path.join(PRODUCTION_DIR, f'scaler_{predictor}_BEST.pkl')
        )


def main():
    print("=" * 70)
    print("DRIVER PROFILER - MODEL EVALUATION")
    print("=" * 70)
    if not TF_AVAILABLE:
        print("\nNOTE: tensorflow not installed — LSTM candidates (if trained) can't be scored here.")

    meta = load_metadata()

    sector_results = evaluate_sector(meta)
    consistency_results, consistency_best = evaluate_consistency(meta)
    style_results, style_best = evaluate_driving_style(meta)
    telemetry_results, telemetry_best = evaluate_telemetry(meta)

    selected = {
        'sector1': sector_results['sector1']['best'],
        'sector2': sector_results['sector2']['best'],
        'sector3': sector_results['sector3']['best'],
        'consistency': consistency_best,
        'style': style_best,
        'telemetry': telemetry_best,
    }
    with open(os.path.join(PRODUCTION_DIR, 'SELECTED_MODELS.json'), 'w') as f:
        json.dump(selected, f, indent=2)

    report = {
        'sector': {k: v['scores'] for k, v in sector_results.items()},
        'consistency': consistency_results,
        'style': style_results,
        'telemetry': telemetry_results,
        'selected_models': selected,
    }
    with open(os.path.join(REPORT_DIR, 'evaluation_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=float)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nSelected models:")
    for k, v in selected.items():
        print(f"   {k}: {v}")
    print(f"\nProduction models in: {PRODUCTION_DIR}")
    print(f"Report + SHAP plots in: {REPORT_DIR}\n")


if __name__ == '__main__':
    main()
