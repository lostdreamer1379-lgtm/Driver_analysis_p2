import React from 'react';

const RATING_COLOR = {
  'very consistent': 'var(--teal)',
  'consistent': 'var(--teal)',
  'variable': 'var(--amber)',
};

const RATING_BADGE = {
  'very consistent': 'badge-low',
  'consistent': 'badge-low',
  'variable': 'badge-moderate',
};

export default function ConsistencyCard({ data }) {
  if (!data) return <ReadoutSkeleton title="CONSISTENCY" />;

  const { std_dev_s, rating } = data;
  const pct = Math.min(100, (std_dev_s / 0.8) * 100);

  return (
    <div className="readout" style={{ '--bar-color': RATING_COLOR[rating] }}>
      <div className="readout-head">
        <span className="readout-title">CONSISTENCY</span>
        <span className={`readout-badge ${RATING_BADGE[rating]}`}>{rating}</span>
      </div>
      <div className="readout-value tabular">
        ±{std_dev_s.toFixed(3)}<span className="unit">s</span>
      </div>
      <p className="readout-sub">Lap-time standard deviation over the last 5 laps.</p>
      <div className="readout-bar">
        <div className="readout-bar-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function ReadoutSkeleton({ title }) {
  return (
    <div className="readout">
      <div className="readout-head"><span className="readout-title">{title}</span></div>
      <div className="skel" style={{ height: 38, width: '55%', marginBottom: 8 }} />
      <div className="skel" style={{ height: 12, width: '80%' }} />
    </div>
  );
}
