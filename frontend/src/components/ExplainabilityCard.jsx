import React from 'react';

function FactorColumn({ title, factors }) {
  if (!factors || factors.length === 0) {
    return (
      <div>
        <p className="factor-col-title">{title}</p>
        <p style={{ fontSize: 12, color: 'var(--text-faint)' }}>No explainability data for this run.</p>
      </div>
    );
  }

  const maxAbs = Math.max(...factors.map((f) => Math.abs(f.impact)), 0.0001);

  return (
    <div>
      <p className="factor-col-title">{title}</p>
      {factors.map((f) => {
        const widthPct = (Math.abs(f.impact) / maxAbs) * 50;
        const isPos = f.impact >= 0;
        return (
          <div className="factor-item" key={f.feature}>
            <span className="factor-name" title={f.feature}>{f.feature}</span>
            <div className="factor-bar-track">
              <div className="mid-line" />
              <div className={`factor-bar-fill ${isPos ? 'pos' : 'neg'}`} style={{ width: `${widthPct}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function ExplainabilityCard({ predictions }) {
  if (!predictions) return null;

  return (
    <div className="panel">
      <div className="panel-head">
        <h2>Why the model says this</h2>
        <span className="hint">SHAP · red pushes up, teal pushes down</span>
      </div>
      <div className="factor-grid">
        <FactorColumn title="SECTOR 1 TIME" factors={predictions.sectors.sector1.shap_factors} />
        <FactorColumn title="CONSISTENCY" factors={predictions.consistency.shap_factors} />
        <FactorColumn title="DRIVING STYLE" factors={predictions.driving_style.shap_factors} />
      </div>
    </div>
  );
}
