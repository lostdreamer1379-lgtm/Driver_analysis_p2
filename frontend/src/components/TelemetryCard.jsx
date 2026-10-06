import React from 'react';

export default function TelemetryCard({ data }) {
  if (!data) return <ReadoutSkeleton />;

  const { seconds_vs_field_avg, faster_than_field } = data;
  const abs = Math.abs(seconds_vs_field_avg);
  const pct = Math.min(100, (abs / 2) * 100);

  return (
    <div className="readout" style={{ '--bar-color': faster_than_field ? 'var(--teal)' : 'var(--accent)' }}>
      <div className="readout-head">
        <span className="readout-title">VS FIELD AVERAGE</span>
        <span className={`readout-badge ${faster_than_field ? 'badge-low' : 'badge-high'}`}>
          {faster_than_field ? 'faster' : 'slower'}
        </span>
      </div>
      <div className="readout-value tabular">
        {faster_than_field ? '-' : '+'}{abs.toFixed(3)}<span className="unit">s/lap</span>
      </div>
      <p className="readout-sub">Modeled pace delta against the field average this lap.</p>
      <div className="readout-bar">
        <div className="readout-bar-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function ReadoutSkeleton() {
  return (
    <div className="readout">
      <div className="readout-head"><span className="readout-title">VS FIELD AVERAGE</span></div>
      <div className="skel" style={{ height: 38, width: '55%', marginBottom: 8 }} />
      <div className="skel" style={{ height: 12, width: '80%' }} />
    </div>
  );
}
