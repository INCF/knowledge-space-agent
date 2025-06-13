import React from 'react';

const Logo: React.FC<{ className?: string }> = ({ className = "" }) => {
  return (
    <svg 
      width="180" 
      height="50" 
      viewBox="0 0 180 50" 
      className={className}
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <linearGradient id="logoGradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#4A90E2" />
          <stop offset="100%" stopColor="#357ABD" />
        </linearGradient>
        <linearGradient id="textGradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#2D3748" />
          <stop offset="100%" stopColor="#4A5568" />
        </linearGradient>
      </defs>
      
      <circle cx="12" cy="15" r="3" fill="url(#logoGradient)" />
      
      <text 
        x="22" 
        y="23" 
        fontFamily="Inter, -apple-system, BlinkMacSystemFont, sans-serif" 
        fontSize="20" 
        fontWeight="700" 
        fill="url(#textGradient)"
        letterSpacing="-0.5px"
      >
        incf
      </text>
      
      <rect x="70" y="8" width="2" height="34" fill="url(#logoGradient)" rx="1" />
      
      <text 
        x="78" 
        y="18" 
        fontFamily="Inter, sans-serif" 
        fontSize="11" 
        fill="#6B7280"
        fontWeight="400"
      >
        enabling open and
      </text>
      
      <text 
        x="78" 
        y="32" 
        fontFamily="Inter, sans-serif" 
        fontSize="11" 
        fill="#6B7280"
        fontWeight="600"
      >
        FAIR neuroscience
      </text>
    </svg>
  );
};

export default Logo;