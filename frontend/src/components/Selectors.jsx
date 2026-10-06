import React from 'react';

const CIRCUITS = [
  'Bahrain', 'Jeddah', 'Melbourne', 'Baku', 'Miami', 'Imola', 'Monaco',
  'Barcelona', 'Montreal', 'Silverstone', 'Budapest', 'Spa', 'Zandvoort',
  'Monza', 'Singapore', 'Suzuka', 'Austin', 'Mexico City', 'São Paulo',
  'Las Vegas', 'Abu Dhabi', 'Lusail',
];

const COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'];

const DRIVERS = [
  'VER', 'HAM', 'LEC', 'NOR', 'PIA', 'SAI', 'RUS', 'ALO', 'STR', 'GAS',
  'OCO', 'TSU', 'ALB', 'HUL', 'BOT', 'ZHO', 'MAG', 'RIC', 'SAR', 'PER',
];

export default function Selectors({ form, setForm, onRun, loading }) {
  const update = (key, value) => setForm((f) => ({ ...f, [key]: value }));

  return (
    <div className="rail">
      <div className="rail-section">
        <p className="rail-label">DRIVER</p>
        <div className="select-wrap">
          <select value={form.driver} onChange={(e) => update('driver', e.target.value)}>
            {DRIVERS.map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="rail-section">
        <p className="rail-label">CIRCUIT</p>
        <div className="select-wrap">
          <select value={form.circuit} onChange={(e) => update('circuit', e.target.value)}>
            {CIRCUITS.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="rail-section">
        <p className="rail-label">COMPOUND</p>
        <div className="compound-grid">
          {COMPOUNDS.map((c) => (
            <button
              key={c}
              className={`compound-chip ${form.compound === c ? `active-${c}` : ''}`}
              onClick={() => update('compound', c)}
              type="button"
            >
              {c.slice(0, 4)}
            </button>
          ))}
        </div>
      </div>

      <div className="rail-section">
        <div className="slider-row">
          <p className="rail-label">TYRE LIFE</p>
          <span className="slider-value tabular">{form.tyre_life} laps</span>
        </div>
        <input
          type="range" min="1" max="40" value={form.tyre_life}
          onChange={(e) => update('tyre_life', Number(e.target.value))}
        />
      </div>

      <div className="rail-section">
        <div className="slider-row">
          <p className="rail-label">AVG SPEED</p>
          <span className="slider-value tabular">{form.avg_speed} km/h</span>
        </div>
        <input
          type="range" min="150" max="260" value={form.avg_speed}
          onChange={(e) => update('avg_speed', Number(e.target.value))}
        />
      </div>

      <div className="rail-section">
        <div className="slider-row">
          <p className="rail-label">MAX SPEED</p>
          <span className="slider-value tabular">{form.max_speed} km/h</span>
        </div>
        <input
          type="range" min="250" max="360" value={form.max_speed}
          onChange={(e) => update('max_speed', Number(e.target.value))}
        />
      </div>

      <div className="rail-section">
        <div className="slider-row">
          <p className="rail-label">AVG THROTTLE</p>
          <span className="slider-value tabular">{form.avg_throttle_pct}%</span>
        </div>
        <input
          type="range" min="20" max="100" value={form.avg_throttle_pct}
          onChange={(e) => update('avg_throttle_pct', Number(e.target.value))}
        />
      </div>

      <div className="rail-section">
        <div className="slider-row">
          <p className="rail-label">BRAKE USAGE</p>
          <span className="slider-value tabular">{form.brake_pct}%</span>
        </div>
        <input
          type="range" min="2" max="35" value={form.brake_pct}
          onChange={(e) => update('brake_pct', Number(e.target.value))}
        />
      </div>

      <button className="run-btn" onClick={onRun} disabled={loading}>
        {loading ? (
          'ANALYZING…'
        ) : (
          <>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="5 3 19 12 5 21 5 3" />
            </svg>
            RUN PROFILE
          </>
        )}
      </button>
    </div>
  );
}
