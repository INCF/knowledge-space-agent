import React, { useState, useEffect, useRef } from 'react';
import Logo from './components/Logo';
import LoadingMessage from './components/LoadingMessage';
import MessageBubble from './components/MessageBubble';

interface Message {
  id: string;
  type: 'user' | 'ai' | 'error';
  content: string;
  timestamp: Date;
}

interface ChatResponse {
  response: string;
  error: boolean;
}

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isOnline, setIsOnline] = useState(true);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Add welcome message
    const welcomeMessage: Message = {
      id: 'welcome',
      type: 'ai',
      content: `Hello! I'm your INCF KnowledgeSpace assistant. I can help you discover and explore neuroscience datasets.

Try asking me something like:
• "Brain imaging"
• "EEG data "
• "fMRI"
• "cognitive neuroscience"`,
      timestamp: new Date()
    };
    setMessages([welcomeMessage]);

    // Check API health
    checkApiHealth();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const checkApiHealth = async () => {
    try {
      const response = await fetch('/api/health');
      setIsOnline(response.ok);
    } catch (error) {
      setIsOnline(false);
    }
  };

  const scrollToBottom = () => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  };

  const formatTime = (date: Date): string => {
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit',
      hour12: true 
    });
  };

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const query = inputValue.trim();
    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: query,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: ChatResponse = await response.json();

      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: data.error ? 'error' : 'ai',
        content: data.response,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'error',
        content: 'Sorry, I encountered an error while processing your request. Please check your connection and try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages(messages.filter(msg => msg.id === 'welcome'));
  };

  return (
    <div className="app-container">
      <div className="main-layout">
        {/* Header */}
        <header className="header-section">
          <div className="header-content">
            <div className="brand-section">
              <Logo className="brand-logo" />
              <div className="brand-text">
                <h1 className="brand-title">KnowledgeSpace AI</h1>
                <p className="brand-subtitle">Neuroscience Dataset Search Assistant</p>
              </div>
            </div>
            <div className="header-actions">
              <div className="status-indicator">
                <div className={`status-dot ${isOnline ? 'online' : 'offline'}`}></div>
                <span className="status-text">{isOnline ? 'Connected' : 'Disconnected'}</span>
              </div>
              <button 
                className="action-btn clear-btn" 
                onClick={clearChat}
                title="Clear conversation"
              >
                <i className="fas fa-broom"></i>
              </button>
            </div>
          </div>
        </header>

        {/* Chat Messages Container */}
        <main 
          className="chat-container" 
          ref={chatContainerRef}
        >
          <div className="messages-wrapper">
            {messages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                formatTime={formatTime}
              />
            ))}
            
            {/* Loading indicator */}
            {isLoading && <LoadingMessage />}
          </div>
        </main>

        {/* Input Area */}
        <footer className="input-section">
          <div className="input-container">
            <div className="input-wrapper">
              <input 
                type="text" 
                className="message-input" 
                placeholder="Ask about neuroscience datasets, brain imaging data, or research topics..."
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                disabled={isLoading}
              />
              <button 
                className={`send-button ${isLoading || !inputValue.trim() ? 'disabled' : ''}`}
                type="button" 
                onClick={sendMessage} 
                disabled={isLoading || !inputValue.trim()}
              >
                {isLoading ? (
                  <i className="fas fa-spinner fa-spin"></i>
                ) : (
                  <i className="fas fa-paper-plane"></i>
                )}
              </button>
            </div>
            <div className="input-footer">
              <i className="fas fa-info-circle"></i>
              <span>Powered by INCF KnowledgeSpace API - Neuroscience datasets</span>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default App;
