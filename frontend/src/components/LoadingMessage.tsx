import React from 'react';

const LoadingMessage: React.FC = () => {
  return (
    <div className="message-bubble ai-message loading-message">
      <div className="message-header">
        <div className="d-flex align-items-center">
          <i className="fas fa-robot text-primary me-2"></i>
          <strong>KnowledgeSpace Assistant</strong>
        </div>
      </div>
      <div className="message-content">
        <div className="loading-container">
          <div className="loading-text">
            <span className="loading-stage">Searching neuroscience datasets</span>
            <div className="loading-dots">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
          <div className="loading-progress">
            <div className="progress-bar"></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoadingMessage;