import React from 'react';

export default function SectorCard({ sectors, totalLap }) {
  if (!sectors) return <SectorSkeleton />;

  return (
    <div className="panel">
      <div className="panel-head">
        <h2>Sector breakdown</h2>
        <span className="hint tabular">lap est. {totalLap?.toFixed(3)}s</span>
      </div>
      <div className="sector-timeline">
        {['sector1', 'sector2', 'sector3'].map((key, i) => (
          <div className="sector-segment" key={key}>
            <div className="seg-label">SECTOR {i + 1}</div>
            <div className="seg-value tabular">{sectors[key].time_s.toFixed(3)}s</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function SectorSkeleton() {
  return (
    <div className="panel">
      <div className="panel-head"><h2>Sector breakdown</h2></div>
      <div className="sector-timeline">
        {[1, 2, 3].map((i) => (
          <div className="sector-segment" key={i}>
            <div className="skel" style={{ height: 10, width: '60%', margin: '0 auto 8px' }} />
            <div className="skel" style={{ height: 17, width: '70%', margin: '0 auto' }} />
          </div>
        ))}
      </div>
    </div>
  );
}
