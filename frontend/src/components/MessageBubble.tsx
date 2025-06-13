import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface Message {
  id: string;
  type: 'user' | 'ai' | 'error';
  content: string;
  timestamp: Date;
}

interface MessageBubbleProps {
  message: Message;
  formatTime: (date: Date) => string;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message, formatTime }) => {
  const renderContent = () => {
    if (message.type === 'ai') {
      return (
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {message.content}
        </ReactMarkdown>
      );
    }
    return <div className="message-text">{message.content}</div>;
  };

  return (
    <div
      className={`message-bubble ${
        message.type === 'user' ? 'user-message' : 
        message.type === 'error' ? 'error-message' : 'ai-message'
      }`}
    >
      <div className="message-header">
        <div className="d-flex align-items-center">
          <i className={`fas ${
            message.type === 'user' ? 'fa-user' : 
            message.type === 'error' ? 'fa-exclamation-triangle' : 'fa-robot text-primary'
          } me-2`}></i>
          <strong>
            {message.type === 'user' ? 'You' : 
             message.type === 'error' ? 'Error' : 'KnowledgeSpace Assistant'}
          </strong>
        </div>
        <small className={`timestamp ${message.type === 'user' ? 'text-light' : 'text-muted'}`}>
          {formatTime(message.timestamp)}
        </small>
      </div>
      <div className="message-content">
        {renderContent()}
      </div>
    </div>
  );
};

export default MessageBubble;