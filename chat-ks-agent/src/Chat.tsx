import React, { useState } from 'react';
import './Chat.css';

export interface ChatMessage {
  sender: 'user' | 'agent';
  text: string;
}

const Chat: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  // To be Replaced with actual KS API endpoint
  const KS_API_URL = 'https://api.example.com/ks-query';
  // To be Replaced with LLM API endpoint
  const LLM_API_URL = 'https://api.example.com/llm-parse';

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    const userMessage: ChatMessage = { sender: 'user', text: input };
    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);
    setInput('');
    try {
      // 1. Query KS API
      const ksRes = await fetch(KS_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: input }),
      });
      const ksData = await ksRes.json();
      // 2. Send KS API result to LLM for parsing
      const llmRes = await fetch(LLM_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ksResult: ksData }),
      });
      const llmData = await llmRes.json();
      const agentMessage: ChatMessage = { sender: 'agent', text: llmData.result || 'No response.' };
      setMessages((prev) => [...prev, agentMessage]);
    } catch (err) {
      setMessages((prev) => [...prev, { sender: 'agent', text: 'Error querying the API.' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`chat-message ${msg.sender}`}>{msg.text}</div>
        ))}
        {loading && <div className="chat-message agent">Thinking...</div>}
      </div>
      <form className="chat-input" onSubmit={sendMessage}>
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Type your question..."
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>Send</button>
      </form>
    </div>
  );
};

export default Chat;
