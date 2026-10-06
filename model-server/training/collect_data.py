"""
DRIVER PROFILER - Data Collection
Pulls per-lap and per-sector data from FastF1, plus a coarse telemetry
summary (speed/throttle/brake stats) per lap, for building:
  - Telemetry Comparison  (speed delta vs field average by segment)
  - Sector Breakdown      (3 sector times)
  - Driver Stats          (driving-style classification features)
  - Consistency Scorer    (lap-time variance over a stint)

Cross-platform thread-based timeout (same approach as Strategy Master),
resumable via a progress.json checkpoint keyed on season|race, and rows
are appended to the output CSV incrementally so nothing is lost on a
crash or Ctrl+C. Re-run the script any time — it only fetches what's
still missing. Pass --fresh to wipe the checkpoint and start over.
"""

import fastf1
import pandas as pd
import numpy as np
import warnings
import os
import sys
import json
import threading

warnings.filterwarnings('ignore')

# --- Paths ---
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, '../../data/cache')
RAW_DIR = os.path.join(HERE, '../../data/raw')
PROCESSED_DIR = os.path.join(HERE, '../../data/processed')
OUT_PATH = os.path.join(RAW_DIR, 'driver_profiler_data.csv')
PROGRESS_PATH = os.path.join(RAW_DIR, 'progress.json')

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# --- Configuration ---
SEASONS = [2023, 2024, 2025, 2026]
MAX_RACES_PER_SEASON = 24
SESSION_TIMEOUT_SEC = 90

with open(os.path.join(HERE, 'circuits_config.json'), 'r') as f:
    CIRCUITS_MAP = json.load(f)

DEFAULT_CIRCUIT = {'corner_ratio': 0.55, 'abrasiveness': 7, 'speed_index': 6, 'lap_length_km': 5.0, 'n_corners': 16}


# --- Thread-based timeout (cross-platform) ---
class TimeoutException(Exception):
    pass


class TimedFunction:
    def __init__(self, timeout_sec=60):
        self.timeout_sec = timeout_sec
        self.result = None
        self.exception = None

    def run(self, func, *args, **kwargs):
        def target():
            try:
                self.result = func(*args, **kwargs)
            except Exception as e:
                self.exception = e

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout_sec)

        if thread.is_alive():
            raise TimeoutException(f"Operation timed out after {self.timeout_sec}s")
        if self.exception:
            raise self.exception
        return self.result


# --- Checkpoint (resume support) ---
def _session_key(season, race_name):
    return f"{season}|{race_name}"


def load_progress():
    if os.path.exists(PROGRESS_PATH):
        try:
            with open(PROGRESS_PATH, 'r') as f:
                data = json.load(f)
            data.setdefault('done', {})
            data.setdefault('failed', {})
            return data
        except Exception:
            pass
    return {'done': {}, 'failed': {}}


def save_progress(progress):
    with open(PROGRESS_PATH, 'w') as f:
        json.dump(progress, f, indent=2)


def append_rows_to_csv(rows, out_path):
    if not rows:
        return
    chunk = pd.DataFrame(rows)
    write_header = not os.path.exists(out_path)
    chunk.to_csv(out_path, mode='a', header=write_header, index=False)


def collect_session(season, race_name):
    """Load one race session and build per-driver-per-lap rows with sector
    times and a coarse telemetry summary. Returns [] on any failure
    (never raises) so one bad race never kills the whole run."""
    try:
        print(f"    {race_name:<24}", end=" ", flush=True)

        try:
            timer = TimedFunction(timeout_sec=SESSION_TIMEOUT_SEC)

            def load_func():
                session = fastf1.get_session(season, race_name, 'Race')
                # telemetry=True here — we need speed/throttle/brake channels,
                # unlike Strategy Master which only needed lap/weather data.
                session.load(telemetry=True, weather=False, messages=False)
                return session

            session = timer.run(load_func)
        except TimeoutException:
            print("TIMEOUT (skipped)")
            return []
        except Exception as e:
            print(f"load error: {str(e)[:40]} (skipped)")
            return []

        laps = session.laps.copy()
        laps = laps[laps['LapTime'].notna()].reset_index(drop=True)
        if len(laps) == 0:
            print("no valid laps")
            return []

        location = session.event.get('Location', race_name)
        circuit_key = next(
            (k for k in CIRCUITS_MAP if k.lower() in str(location).lower() or str(location).lower() in k.lower()),
            None
        )
        circ = CIRCUITS_MAP.get(circuit_key, DEFAULT_CIRCUIT)

        rows = []
        for driver in laps['Driver'].unique():
            d_laps = laps[laps['Driver'] == driver].reset_index(drop=True)
            if len(d_laps) < 3:
                continue  # not enough laps for a meaningful sequence

            lap_times = d_laps['LapTime'].dt.total_seconds().values
            field_avg_lap = laps['LapTime'].dt.total_seconds().mean()

            for idx, lap in d_laps.iterrows():
                lap_time_s = lap['LapTime'].total_seconds()

                s1 = lap.get('Sector1Time')
                s2 = lap.get('Sector2Time')
                s3 = lap.get('Sector3Time')
                s1_s = s1.total_seconds() if pd.notna(s1) else lap_time_s * 0.28
                s2_s = s2.total_seconds() if pd.notna(s2) else lap_time_s * 0.37
                s3_s = s3.total_seconds() if pd.notna(s3) else lap_time_s * 0.35

                # Coarse telemetry summary for this lap (fast channels, not full trace,
                # to keep the dataset a manageable size for a from-scratch LSTM)
                try:
                    tel = lap.get_car_data()
                    avg_speed = float(tel['Speed'].mean()) if len(tel) else np.nan
                    max_speed = float(tel['Speed'].max()) if len(tel) else np.nan
                    avg_throttle = float(tel['Throttle'].mean()) if 'Throttle' in tel.columns and len(tel) else np.nan
                    brake_pct = float(tel['Brake'].mean()) * 100 if 'Brake' in tel.columns and len(tel) else np.nan
                except Exception:
                    avg_speed, max_speed, avg_throttle, brake_pct = np.nan, np.nan, np.nan, np.nan

                speed_delta_vs_field = (lap_time_s - field_avg_lap) * -1  # positive = faster than field

                rows.append({
                    'season': season, 'race': race_name, 'circuit': circuit_key or race_name,
                    'driver': driver, 'lap_number': int(lap['LapNumber']),
                    'lap_time_s': round(lap_time_s, 4),
                    'sector1_s': round(s1_s, 4), 'sector2_s': round(s2_s, 4), 'sector3_s': round(s3_s, 4),
                    'avg_speed': round(avg_speed, 2) if pd.notna(avg_speed) else None,
                    'max_speed': round(max_speed, 2) if pd.notna(max_speed) else None,
                    'avg_throttle_pct': round(avg_throttle, 2) if pd.notna(avg_throttle) else None,
                    'brake_pct': round(brake_pct, 2) if pd.notna(brake_pct) else None,
                    'speed_delta_vs_field': round(speed_delta_vs_field, 4),
                    'corner_ratio': circ['corner_ratio'], 'n_corners': circ['n_corners'],
                    'abrasiveness': circ['abrasiveness'], 'speed_index': circ['speed_index'],
                    'lap_length_km': circ['lap_length_km'],
                    'compound': str(lap.get('Compound', 'UNKNOWN')).upper(),
                    'tyre_life': int(lap.get('TyreLife', 1)) if pd.notna(lap.get('TyreLife')) else 1,
                    'track_position': int(lap.get('Position')) if pd.notna(lap.get('Position')) else None,
                })

        print(f"ok ({len(rows)} laps, {laps['Driver'].nunique()} drivers)")
        return rows

    except Exception as e:
        print(f"error: {str(e)[:50]}")
        return []


def run_collection(fresh=False):
    if fresh:
        for p in (OUT_PATH, PROGRESS_PATH):
            if os.path.exists(p):
                os.remove(p)

    progress = load_progress()
    newly_done, newly_failed, skipped = 0, 0, 0

    print("\n" + "=" * 70)
    print("DRIVER PROFILER - FASTF1 DATA COLLECTION")
    print("=" * 70)
    if progress['done']:
        prior_rows = sum(progress['done'].values())
        print(f"\nResuming — {len(progress['done'])} session(s) already collected "
              f"({prior_rows:,} rows). Use --fresh to ignore and start over.")

    for season in SEASONS:
        print(f"\nSeason {season}")
        print("-" * 70)
        try:
            schedule = fastf1.get_event_schedule(season, include_testing=False)
            races = schedule['EventName'].tolist()[:MAX_RACES_PER_SEASON]
            print(f"   Found {len(races)} races\n")
        except Exception as e:
            print(f"   Schedule error: {e}")
            continue

        for race in races:
            key = _session_key(season, race)

            if key in progress['done']:
                print(f"    {race:<24} already collected, skipping "
                      f"({progress['done'][key]} laps)")
                skipped += 1
                continue

            result = collect_session(season, race)
            if result:
                append_rows_to_csv(result, OUT_PATH)
                progress['done'][key] = len(result)
                progress['failed'].pop(key, None)
                newly_done += 1
            else:
                progress['failed'][key] = 'load failed or no data — will retry next run'
                newly_failed += 1

            save_progress(progress)

    total_rows = sum(progress['done'].values())
    if total_rows == 0:
        print("\nNO DATA COLLECTED — check network access / FastF1 availability.")
        return

    df = pd.read_csv(OUT_PATH)

    print("\n" + "=" * 70)
    print("COLLECTION COMPLETE")
    print("=" * 70)
    print(f"\nThis run:")
    print(f"   Newly collected: {newly_done}")
    print(f"   Newly failed:    {newly_failed}")
    print(f"   Skipped (done):  {skipped}")
    print(f"\nOverall:")
    print(f"   Sessions done:   {len(progress['done'])}")
    print(f"   Sessions failed: {len(progress['failed'])} (will retry on next run)")
    print(f"   Total records:   {len(df):,}")
    print(f"   Circuits:        {df['circuit'].nunique()}")
    print(f"   Drivers:         {df['driver'].nunique()}")
    print(f"\nSaved to: {OUT_PATH}")
    print(f"Progress checkpoint: {PROGRESS_PATH}\n")


if __name__ == '__main__':
    run_collection(fresh='--fresh' in sys.argv)
