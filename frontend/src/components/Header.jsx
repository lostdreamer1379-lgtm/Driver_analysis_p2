import React from 'react';
import ThemeToggle from './ThemeToggle';

export default function Header({ theme, onToggleTheme }) {
  return (
    <header className="tower-header">
      <div className="brand">
        <div className="brand-mark">
          <svg viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="3" />
            <path d="M12 2v4M12 18v4M22 12h-4M6 12H2M19.07 4.93l-2.83 2.83M7.76 16.24l-2.83 2.83M19.07 19.07l-2.83-2.83M7.76 7.76L4.93 4.93" />
          </svg>
        </div>
        <div className="brand-text">
          <h1>Driver Profiler</h1>
          <p>telemetry · sector pace · consistency · style</p>
        </div>
      </div>
      <div className="header-right">
        <div className="live-pill">
          <span className="live-dot" />
          MODEL ONLINE
        </div>
        <ThemeToggle theme={theme} onToggle={onToggleTheme} />
      </div>
    </header>
  );
}
