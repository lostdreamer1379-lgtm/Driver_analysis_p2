import React from 'react';

const STYLE_META = {
  aggressive: { icon: '🔴', sub: 'Late braking, high throttle aggression' },
  conservative: { icon: '🟢', sub: 'Smooth inputs, tyre-conscious pace' },
  flowing: { icon: '🔵', sub: 'High minimum speeds through corners' },
  balanced: { icon: '🟡', sub: 'No strong bias either way' },
};

export default function StyleCard({ data }) {
  if (!data) return <ReadoutSkeleton />;

  const { label } = data;
  const meta = STYLE_META[label] || { icon: '⚪', sub: '' };

  return (
    <div className="readout">
      <div className="readout-head">
        <span className="readout-title">DRIVING STYLE</span>
      </div>
      <div className="style-badge-wrap">
        <div className={`style-badge style-${label}`}>{meta.icon}</div>
        <div className="style-text">
          <div className="style-label">{label}</div>
          <div className="style-sub">{meta.sub}</div>
        </div>
      </div>
    </div>
  );
}

function ReadoutSkeleton() {
  return (
    <div className="readout">
      <div className="readout-head"><span className="readout-title">DRIVING STYLE</span></div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <div className="skel" style={{ width: 52, height: 52, borderRadius: '50%' }} />
        <div style={{ flex: 1 }}>
          <div className="skel" style={{ height: 15, width: '50%', marginBottom: 6 }} />
          <div className="skel" style={{ height: 12, width: '80%' }} />
        </div>
      </div>
    </div>
  );
}
