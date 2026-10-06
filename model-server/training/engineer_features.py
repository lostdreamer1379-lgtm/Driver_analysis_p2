"""
DRIVER PROFILER - Feature Engineering
Builds two parallel representations from the raw per-lap data:
  1. Flattened features (for XGBoost / Random Forest) — one row per lap
     with rolling stats, encodings, interaction terms.
  2. Sequences (for the LSTM) — fixed-length windows of consecutive laps
     per driver-stint, saved as a numpy array alongside the flattened CSV.

Targets:
  - sector_breakdown: predict sector1/2/3 time (regression, 3 outputs handled
    as 3 separate models sharing the same feature set)
  - consistency: rolling stdev of lap time over the last 5 laps (regression)
  - driving_style: KMeans-derived cluster label from throttle/brake/speed
    aggression metrics (classification target, built here since FastF1 has
    no ground-truth "driving style" label)
  - telemetry_delta: speed_delta_vs_field (regression)
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(HERE, '../../data/raw/driver_profiler_data.csv')
OUT_PATH = os.path.join(HERE, '../../data/processed/driver_profiler_features.csv')
SEQ_PATH = os.path.join(HERE, '../../data/processed/sequences.npz')
ENCODERS_PATH = os.path.join(HERE, '../models/label_encoders.pkl')
METADATA_PATH = os.path.join(HERE, '../models/model_metadata.json')
KMEANS_PATH = os.path.join(HERE, '../models/driving_style_kmeans.pkl')

SEQ_LEN = 5  # laps of history fed to the LSTM per prediction


def engineer():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Raw data not found at {RAW_PATH}. Run collect_data.py first.")

    print("Loading raw data...")
    df = pd.read_csv(RAW_PATH)
    print(f"   {len(df):,} rows loaded")

    # Drop rows with no telemetry summary at all (can't build style features)
    telemetry_cols = ['avg_speed', 'max_speed', 'avg_throttle_pct', 'brake_pct']
    df = df.dropna(subset=['lap_time_s', 'sector1_s', 'sector2_s', 'sector3_s']).reset_index(drop=True)
    for c in telemetry_cols:
        df[c] = df[c].fillna(df[c].median())

    # --- Encode categoricals ---
    print("Encoding categorical features...")
    encoders = {}
    for col in ['circuit', 'compound', 'driver']:
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    os.makedirs(os.path.dirname(ENCODERS_PATH), exist_ok=True)
    joblib.dump(encoders, ENCODERS_PATH)

    df = df.sort_values(['season', 'race', 'driver', 'lap_number']).reset_index(drop=True)
    group_keys = ['season', 'race', 'driver']

    # --- Rolling / consistency features ---
    print("Computing rolling consistency features...")
    df['lap_time_roll5_std'] = (
        df.groupby(group_keys)['lap_time_s']
        .transform(lambda s: s.rolling(5, min_periods=2).std())
    )
    df['lap_time_roll3_mean'] = (
        df.groupby(group_keys)['lap_time_s']
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )
    df['sector_sum_check'] = df['sector1_s'] + df['sector2_s'] + df['sector3_s']

    # --- Driving-style clustering (unsupervised target for the classifier) ---
    print("Deriving driving-style clusters...")
    style_features = df[['avg_throttle_pct', 'brake_pct', 'max_speed']].fillna(
        df[['avg_throttle_pct', 'brake_pct', 'max_speed']].median()
    )
    style_scaler = StandardScaler()
    style_scaled = style_scaler.fit_transform(style_features)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['driving_style'] = kmeans.fit_predict(style_scaled)
    os.makedirs(os.path.dirname(KMEANS_PATH), exist_ok=True)
    joblib.dump({'kmeans': kmeans, 'scaler': style_scaler}, KMEANS_PATH)

    # Label the clusters by their centroid characteristics for readability
    centroids = style_scaler.inverse_transform(kmeans.cluster_centers_)
    style_names = {}
    for i, (throttle, brake, speed) in enumerate(centroids):
        if brake > np.median(centroids[:, 1]) and throttle > np.median(centroids[:, 0]):
            style_names[i] = 'aggressive'
        elif brake < np.median(centroids[:, 1]) and throttle < np.median(centroids[:, 0]):
            style_names[i] = 'conservative'
        elif speed > np.median(centroids[:, 2]):
            style_names[i] = 'flowing'
        else:
            style_names[i] = 'balanced'

    # --- Interaction features ---
    df['corner_x_brake'] = df['corner_ratio'] * df['brake_pct'].fillna(0)
    df['tyre_x_consistency'] = df['tyre_life'] * df['lap_time_roll5_std'].fillna(0)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # --- Feature lists per predictor ---
    sector_features = [
        'circuit_enc', 'compound_enc', 'driver_enc', 'tyre_life',
        'corner_ratio', 'n_corners', 'abrasiveness', 'speed_index', 'lap_length_km',
        'avg_speed', 'max_speed', 'avg_throttle_pct', 'brake_pct',
    ] if 'n_corners' in df.columns else [
        'circuit_enc', 'compound_enc', 'driver_enc', 'tyre_life',
        'corner_ratio', 'abrasiveness', 'speed_index', 'lap_length_km',
        'avg_speed', 'max_speed', 'avg_throttle_pct', 'brake_pct',
    ]
    consistency_features = [
        'circuit_enc', 'compound_enc', 'tyre_life', 'lap_time_roll3_mean',
        'corner_ratio', 'abrasiveness', 'avg_throttle_pct', 'brake_pct',
        'tyre_x_consistency',
    ]
    style_features_list = [
        'avg_speed', 'max_speed', 'avg_throttle_pct', 'brake_pct',
        'corner_ratio', 'corner_x_brake', 'speed_index',
    ]
    telemetry_features = [
        'circuit_enc', 'driver_enc', 'avg_speed', 'max_speed',
        'avg_throttle_pct', 'brake_pct', 'corner_ratio', 'speed_index',
        'tyre_life',
    ]

    metadata = {
        'sector_features': sector_features,
        'consistency_features': consistency_features,
        'style_features': style_features_list,
        'telemetry_features': telemetry_features,
        'targets': {
            'sector1': 'sector1_s', 'sector2': 'sector2_s', 'sector3': 'sector3_s',
            'consistency': 'lap_time_roll5_std',
            'driving_style': 'driving_style',
            'telemetry_delta': 'speed_delta_vs_field',
        },
        'style_names': style_names,
        'seq_len': SEQ_LEN,
        'n_rows': len(df),
        'circuits': sorted(df['circuit'].unique().tolist()),
        'compounds': sorted(df['compound'].unique().tolist()),
        'drivers': sorted(df['driver'].unique().tolist()),
    }
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nFlattened features saved: {OUT_PATH} ({len(df):,} rows)")

    # --- Build fixed-length sequences for the LSTM ---
    print(f"\nBuilding lap sequences (length={SEQ_LEN}) for LSTM...")
    seq_feature_cols = ['lap_time_s', 'sector1_s', 'sector2_s', 'sector3_s',
                         'avg_speed', 'avg_throttle_pct', 'brake_pct', 'tyre_life']
    sequences, seq_targets_consistency, seq_targets_sector1 = [], [], []

    for (season, race, driver), g in df.groupby(['season', 'race', 'driver']):
        g = g.sort_values('lap_number').reset_index(drop=True)
        if len(g) < SEQ_LEN + 1:
            continue
        vals = g[seq_feature_cols].values
        for i in range(len(g) - SEQ_LEN):
            window = vals[i:i + SEQ_LEN]
            target_row = g.iloc[i + SEQ_LEN]
            sequences.append(window)
            seq_targets_consistency.append(target_row['lap_time_roll5_std'])
            seq_targets_sector1.append(target_row['sector1_s'])

    if sequences:
        X_seq = np.array(sequences, dtype=np.float32)
        y_consistency = np.array(seq_targets_consistency, dtype=np.float32)
        y_sector1 = np.array(seq_targets_sector1, dtype=np.float32)
        np.savez_compressed(
            SEQ_PATH, X=X_seq, y_consistency=y_consistency, y_sector1=y_sector1,
            feature_names=np.array(seq_feature_cols)
        )
        print(f"   Sequences saved: {SEQ_PATH} ({X_seq.shape[0]:,} windows, shape {X_seq.shape})")
    else:
        print("   WARNING: not enough consecutive laps per driver-race to build sequences. "
              "LSTM training will be skipped — collect more data first.")

    print(f"\nEncoders: {ENCODERS_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Driving-style KMeans: {KMEANS_PATH}")


if __name__ == '__main__':
    engineer()
