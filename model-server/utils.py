"""
DRIVER PROFILER - Server Utilities
Loads circuit characteristics from the SAME JSON config used by training
(training/circuits_config.json), so there's one source of truth between
the training pipeline and the live API — same pattern as Strategy Master.
"""

import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
CIRCUITS_PATH = os.path.join(HERE, 'training/circuits_config.json')
METADATA_PATH = os.path.join(HERE, 'models/model_metadata.json')

DEFAULT_CIRCUIT = {'corner_ratio': 0.55, 'abrasiveness': 7, 'speed_index': 6, 'lap_length_km': 5.0, 'n_corners': 16}

_circuits_config = None
_metadata = None


def _load_circuits():
    global _circuits_config
    if _circuits_config is None:
        with open(CIRCUITS_PATH, 'r') as f:
            _circuits_config = json.load(f)
    return _circuits_config


def get_metadata():
    global _metadata
    if _metadata is None:
        with open(METADATA_PATH, 'r') as f:
            _metadata = json.load(f)
    return _metadata


def get_circuit_features(circuit):
    circuits = _load_circuits()
    return circuits.get(circuit, DEFAULT_CIRCUIT)


def list_circuits():
    return sorted(_load_circuits().keys())


def encode_label(value, encoder_classes):
    classes = list(encoder_classes)
    try:
        return classes.index(value)
    except ValueError:
        return 0


def build_feature_vector(predictor, payload, encoders, meta):
    """Assemble the ordered numeric feature vector for one predictor.
    payload: normalized request dict (circuit, compound, driver, tyre_life,
             avg_speed, max_speed, avg_throttle_pct, brake_pct, ...)
    """
    circuit = payload['circuit']
    compound = str(payload.get('compound', 'MEDIUM')).upper()
    driver = payload.get('driver', 'VER')
    circ = get_circuit_features(circuit)

    circuit_enc = encode_label(circuit, encoders['circuit'].classes_)
    compound_enc = encode_label(compound, encoders['compound'].classes_)
    driver_enc = encode_label(driver, encoders['driver'].classes_)

    tyre_life = float(payload.get('tyre_life', 10))
    avg_speed = float(payload.get('avg_speed', 210))
    max_speed = float(payload.get('max_speed', 310))
    avg_throttle = float(payload.get('avg_throttle_pct', 65))
    brake_pct = float(payload.get('brake_pct', 12))

    corner_x_brake = circ['corner_ratio'] * brake_pct
    lap_time_roll3_mean = float(payload.get('recent_avg_lap_s', 90))
    lap_time_roll5_std = float(payload.get('recent_std_s', 0.3))
    tyre_x_consistency = tyre_life * lap_time_roll5_std

    feature_values = {
        'circuit_enc': circuit_enc, 'compound_enc': compound_enc, 'driver_enc': driver_enc,
        'tyre_life': tyre_life, 'corner_ratio': circ['corner_ratio'],
        'n_corners': circ.get('n_corners', 16), 'abrasiveness': circ['abrasiveness'],
        'speed_index': circ['speed_index'], 'lap_length_km': circ['lap_length_km'],
        'avg_speed': avg_speed, 'max_speed': max_speed,
        'avg_throttle_pct': avg_throttle, 'brake_pct': brake_pct,
        'lap_time_roll3_mean': lap_time_roll3_mean, 'corner_x_brake': corner_x_brake,
        'tyre_x_consistency': tyre_x_consistency,
    }

    key_map = {
        'sector1': 'sector_features', 'sector2': 'sector_features', 'sector3': 'sector_features',
        'consistency': 'consistency_features', 'style': 'style_features', 'telemetry': 'telemetry_features',
    }
    feature_list = meta[key_map[predictor]]
    return [feature_values[name] for name in feature_list]
