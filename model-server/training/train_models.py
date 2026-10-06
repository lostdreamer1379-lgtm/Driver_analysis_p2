"""
DRIVER PROFILER - Model Training
Trains 3 candidate models for each predictor:
  - Sector Breakdown (regression x3): Linear Regression | XGBoost | LSTM (sequence)
  - Consistency (regression):         Linear Regression | XGBoost | LSTM (sequence)
  - Driving Style (classification):   Logistic Regression | XGBoost | Random Forest
  - Telemetry Delta (regression):     Linear Regression | XGBoost | Random Forest

LSTM is included for the two sequence-shaped targets (sector1, consistency)
since lap-over-lap history is naturally sequential — this is Driver
Profiler's distinguishing model type vs. Strategy Master's tree-only setup.
Keras is optional: if tensorflow isn't installed, LSTM training is skipped
and evaluate_models.py falls back to comparing just the other 2 candidates.
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, '../../data/processed/driver_profiler_features.csv')
SEQ_PATH = os.path.join(HERE, '../../data/processed/sequences.npz')
METADATA_PATH = os.path.join(HERE, '../models/model_metadata.json')
MODELS_DIR = os.path.join(HERE, '../models')

RANDOM_STATE = 42


def load_data_and_metadata():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Run engineer_features.py first.")
    df = pd.read_csv(DATA_PATH)
    with open(METADATA_PATH, 'r') as f:
        meta = json.load(f)
    return df, meta


def _save_sklearn_pair(name_prefix, lr_or_logr, xgb_model, rf_model, scaler=None):
    joblib.dump(lr_or_logr, os.path.join(MODELS_DIR, f'linear_{name_prefix}.pkl'))
    xgb_model.save_model(os.path.join(MODELS_DIR, f'xgb_{name_prefix}.json'))
    joblib.dump(rf_model, os.path.join(MODELS_DIR, f'rf_{name_prefix}.pkl'))
    if scaler is not None:
        joblib.dump(scaler, os.path.join(MODELS_DIR, f'scaler_{name_prefix}.pkl'))


def train_sector(df, features):
    """Trains sector1/2/3 regressors. Linear + XGBoost always; LSTM if TF available
    and sequences exist (sector1 only, since that's what sequences.npz stores)."""
    print("\n[1/4] Sector breakdown (regression x3)")
    for sector in ['sector1', 'sector2', 'sector3']:
        target_col = f'{sector}_s'
        print(f"   -- {sector} --")
        X = df[features].values
        y = df[target_col].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        lr = LinearRegression()
        lr.fit(X_train_s, y_train)

        xgb_model = xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE
        )
        xgb_model.fit(X_train, y_train)

        rf = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train, y_train)

        _save_sklearn_pair(sector, lr, xgb_model, rf, scaler)
        joblib.dump({'X_test': X_test, 'X_test_s': X_test_s, 'y_test': y_test},
                    os.path.join(MODELS_DIR, f'test_split_{sector}.pkl'))
        print(f"      Linear Regression, XGBoost, Random Forest trained")

    # LSTM for sector1 using the real lap sequences, if available
    if TF_AVAILABLE and os.path.exists(SEQ_PATH):
        print("   -- sector1 LSTM (sequence model) --")
        data = np.load(SEQ_PATH)
        X_seq, y_seq = data['X'], data['y_sector1']
        if len(X_seq) >= 50:
            split = int(len(X_seq) * 0.8)
            X_train_seq, X_test_seq = X_seq[:split], X_seq[split:]
            y_train_seq, y_test_seq = y_seq[:split], y_seq[split:]

            lstm = models.Sequential([
                layers.Input(shape=(X_seq.shape[1], X_seq.shape[2])),
                layers.LSTM(32, return_sequences=False),
                layers.Dense(16, activation='relu'),
                layers.Dense(1),
            ])
            lstm.compile(optimizer='adam', loss='mse', metrics=['mae'])
            lstm.fit(X_train_seq, y_train_seq, epochs=20, batch_size=32,
                     validation_split=0.1, verbose=0)

            lstm.save(os.path.join(MODELS_DIR, 'lstm_sector1.keras'))
            joblib.dump({'X_test': X_test_seq, 'y_test': y_test_seq},
                        os.path.join(MODELS_DIR, 'test_split_sector1_lstm.pkl'))
            print("      LSTM trained")
        else:
            print(f"      Skipped — only {len(X_seq)} sequence windows (need >= 50)")
    else:
        reason = "tensorflow not installed" if not TF_AVAILABLE else "no sequences.npz"
        print(f"   -- sector1 LSTM skipped ({reason}) --")


def train_consistency(df, features):
    print("\n[2/4] Consistency scorer (regression)")
    X = df[features].values
    y = df['lap_time_roll5_std'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_s, y_train)

    xgb_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE
    )
    xgb_model.fit(X_train, y_train)

    rf = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)

    _save_sklearn_pair('consistency', lr, xgb_model, rf, scaler)
    joblib.dump({'X_test': X_test, 'X_test_s': X_test_s, 'y_test': y_test},
                os.path.join(MODELS_DIR, 'test_split_consistency.pkl'))
    print("   Linear Regression, XGBoost, Random Forest trained")

    if TF_AVAILABLE and os.path.exists(SEQ_PATH):
        print("   -- consistency LSTM (sequence model) --")
        data = np.load(SEQ_PATH)
        X_seq, y_seq = data['X'], data['y_consistency']
        if len(X_seq) >= 50:
            split = int(len(X_seq) * 0.8)
            X_train_seq, X_test_seq = X_seq[:split], X_seq[split:]
            y_train_seq, y_test_seq = y_seq[:split], y_seq[split:]

            lstm = models.Sequential([
                layers.Input(shape=(X_seq.shape[1], X_seq.shape[2])),
                layers.LSTM(32, return_sequences=False),
                layers.Dense(16, activation='relu'),
                layers.Dense(1),
            ])
            lstm.compile(optimizer='adam', loss='mse', metrics=['mae'])
            lstm.fit(X_train_seq, y_train_seq, epochs=20, batch_size=32,
                     validation_split=0.1, verbose=0)

            lstm.save(os.path.join(MODELS_DIR, 'lstm_consistency.keras'))
            joblib.dump({'X_test': X_test_seq, 'y_test': y_test_seq},
                        os.path.join(MODELS_DIR, 'test_split_consistency_lstm.pkl'))
            print("      LSTM trained")
        else:
            print(f"      Skipped — only {len(X_seq)} sequence windows (need >= 50)")
    else:
        reason = "tensorflow not installed" if not TF_AVAILABLE else "no sequences.npz"
        print(f"   -- consistency LSTM skipped ({reason}) --")


def train_driving_style(df, features):
    print("\n[3/4] Driving style (classification)")
    X = df[features].values
    y = df['driving_style'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler_style.pkl'))

    # multi_class is intentionally omitted: sklearn >=1.5 removed the kwarg
    # (multinomial handling is automatic for multi-class targets with lbfgs,
    # which is the default solver), and older sklearn's 'auto' was already
    # the default too — so omitting it is correct on both.
    logr = LogisticRegression(max_iter=1000)
    logr.fit(X_train_s, y_train)
    joblib.dump(logr, os.path.join(MODELS_DIR, 'logr_style.pkl'))

    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, eval_metric='mlogloss'
    )
    xgb_model.fit(X_train, y_train)
    xgb_model.save_model(os.path.join(MODELS_DIR, 'xgb_style.json'))

    rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(MODELS_DIR, 'rf_style.pkl'))

    joblib.dump({'X_test': X_test, 'X_test_s': X_test_s, 'y_test': y_test},
                os.path.join(MODELS_DIR, 'test_split_style.pkl'))
    print("   Logistic Regression, XGBoost, Random Forest trained")


def train_telemetry_delta(df, features):
    print("\n[4/4] Telemetry delta vs field (regression)")
    X = df[features].values
    y = df['speed_delta_vs_field'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_s, y_train)

    xgb_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE
    )
    xgb_model.fit(X_train, y_train)

    rf = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)

    _save_sklearn_pair('telemetry', lr, xgb_model, rf, scaler)
    joblib.dump({'X_test': X_test, 'X_test_s': X_test_s, 'y_test': y_test},
                os.path.join(MODELS_DIR, 'test_split_telemetry.pkl'))
    print("   Linear Regression, XGBoost, Random Forest trained")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("=" * 70)
    print("DRIVER PROFILER - MODEL TRAINING")
    print("=" * 70)
    if not TF_AVAILABLE:
        print("\nNOTE: tensorflow not installed — LSTM candidates will be skipped.")
        print("      Install with: pip install tensorflow")

    df, meta = load_data_and_metadata()
    print(f"\nLoaded {len(df):,} rows")

    train_sector(df, meta['sector_features'])
    train_consistency(df, meta['consistency_features'])
    train_driving_style(df, meta['style_features'])
    train_telemetry_delta(df, meta['telemetry_features'])

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE — models saved to ../models/")
    print("Run evaluate_models.py next to compare and promote the best.")
    print("=" * 70)


if __name__ == '__main__':
    main()
