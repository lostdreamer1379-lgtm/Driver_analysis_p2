import React, { useEffect, useState, useCallback } from 'react';
import Header from './components/Header';
import Selectors from './components/Selectors';
import SectorCard from './components/SectorCard';
import ConsistencyCard from './components/ConsistencyCard';
import StyleCard from './components/StyleCard';
import TelemetryCard from './components/TelemetryCard';
import ExplainabilityCard from './components/ExplainabilityCard';

const DEFAULT_FORM = {
  circuit: 'Silverstone',
  compound: 'MEDIUM',
  driver: 'VER',
  tyre_life: 10,
  avg_speed: 210,
  max_speed: 310,
  avg_throttle_pct: 65,
  brake_pct: 12,
  recent_avg_lap_s: 90,
  recent_std_s: 0.3,
};

// Deterministic mock so the UI is fully demoable without the HF Spaces
// backend being deployed/warm yet — same pattern as Strategy Master.
function mockPredict(form) {
  const speedFactor = (form.max_speed - 280) / 80;
  const throttleFactor = form.avg_throttle_pct / 100;

  const s1 = 24 + (1 - throttleFactor) * 4 + Math.random() * 0.3;
  const s2 = 31 - speedFactor * 3 + Math.random() * 0.3;
  const s3 = 27 + (form.brake_pct / 35) * 3 + Math.random() * 0.3;
  const totalLap = s1 + s2 + s3;

  const stdDev = Math.max(0.05, 0.5 - throttleFactor * 0.3 + (form.tyre_life / 40) * 0.2);
  const rating = stdDev < 0.15 ? 'very consistent' : stdDev < 0.35 ? 'consistent' : 'variable';

  let label = 'balanced';
  if (form.brake_pct > 20 && throttleFactor > 0.7) label = 'aggressive';
  else if (form.brake_pct < 10 && throttleFactor < 0.6) label = 'conservative';
  else if (form.max_speed > 330) label = 'flowing';

  const delta = (90 - totalLap) * 0.4 + (Math.random() - 0.5) * 0.3;

  const sectorFactors = [
    { feature: 'avg_throttle_pct', impact: (throttleFactor - 0.6) * 1.2 },
    { feature: 'corner_ratio', impact: 0.3 },
    { feature: 'tyre_life', impact: form.tyre_life * 0.015 },
    { feature: 'max_speed', impact: -speedFactor * 0.5 },
    { feature: 'brake_pct', impact: form.brake_pct * 0.02 },
  ];
  const consistencyFactors = [
    { feature: 'tyre_life', impact: form.tyre_life * 0.01 },
    { feature: 'avg_throttle_pct', impact: -(throttleFactor - 0.5) * 0.2 },
    { feature: 'brake_pct', impact: form.brake_pct * 0.01 },
    { feature: 'corner_ratio', impact: 0.05 },
  ];
  const styleFactors = [
    { feature: 'brake_pct', impact: (form.brake_pct - 15) * 0.03 },
    { feature: 'avg_throttle_pct', impact: (throttleFactor - 0.6) * 0.4 },
    { feature: 'max_speed', impact: speedFactor * 0.3 },
  ];

  return {
    success: true,
    input: form,
    predictions: {
      sectors: {
        sector1: { time_s: s1, shap_factors: sectorFactors },
        sector2: { time_s: s2, shap_factors: [] },
        sector3: { time_s: s3, shap_factors: [] },
      },
      estimated_lap_time_s: totalLap,
      consistency: { std_dev_s: stdDev, rating, shap_factors: consistencyFactors },
      driving_style: { label, cluster_id: 0, shap_factors: styleFactors },
      telemetry_delta: { seconds_vs_field_avg: delta, faster_than_field: delta > 0, shap_factors: [] },
    },
    model_sources: { sector1: 'xgboost', consistency: 'xgboost', style: 'xgboost', telemetry: 'xgboost' },
    _mock: true,
  };
}

export default function App() {
  const [theme, setTheme] = useState(() => {
    if (typeof window === 'undefined') return 'dark';
    return window.localStorage.getItem('dp-theme') || 'dark';
  });
  const [form, setForm] = useState(DEFAULT_FORM);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [usingMock, setUsingMock] = useState(false);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    window.localStorage.setItem('dp-theme', theme);
  }, [theme]);

  const toggleTheme = () => setTheme((t) => (t === 'dark' ? 'light' : 'dark'));

  const run = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      if (!res.ok) throw new Error('API unavailable');
      const json = await res.json();
      if (!json.success) throw new Error(json.error || 'Prediction failed');
      setResult(json);
      setUsingMock(false);
    } catch (e) {
      setResult(mockPredict(form));
      setUsingMock(true);
    } finally {
      setLoading(false);
    }
  }, [form]);

  useEffect(() => {
    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="app">
      <div className="shell">
        <Header theme={theme} onToggleTheme={toggleTheme} />

        <div className="layout">
          <Selectors form={form} setForm={setForm} onRun={run} loading={loading} />

          <div className="data-zone">
            {!result ? (
              <EmptyState />
            ) : (
              <>
                <div className="readout-strip">
                  <ConsistencyCard data={result.predictions.consistency} />
                  <StyleCard data={result.predictions.driving_style} />
                  <TelemetryCard data={result.predictions.telemetry_delta} />
                </div>

                <SectorCard sectors={result.predictions.sectors} totalLap={result.predictions.estimated_lap_time_s} />

                <ExplainabilityCard predictions={result.predictions} />
              </>
            )}

            <Footer usingMock={usingMock} sources={result?.model_sources} />
          </div>
        </div>
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="empty-state">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="3" />
        <path d="M12 2v4M12 18v4M22 12h-4M6 12H2" />
      </svg>
      <h3>No profile run yet</h3>
      <p>Pick a driver, circuit and telemetry inputs on the left, then run the profiler.</p>
    </div>
  );
}

function Footer({ usingMock, sources }) {
  return (
    <div className="footer">
      <span>
        {usingMock
          ? 'demo mode — local heuristic (model server not reached)'
          : sources
            ? `models: s1=${sources.sector1} · cons=${sources.consistency} · style=${sources.style}`
            : ''}
      </span>
      <span>Driver Profiler · Project 2 of 6 · F1 Analytics Platform</span>
    </div>
  );
}
