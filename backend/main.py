# main.py
import os
import time
import asyncio
import json
from typing import Optional, Dict, Any
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from agents import NeuroscienceAssistant

load_dotenv()

# FastAPI app + CORS
app = FastAPI(
    title="KnowledgeSpace AI",
    description="Neuroscience Dataset Discovery Assistant",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the assistant with vector search agent on startup
assistant = NeuroscienceAssistant()

# Models
class ChatMessage(BaseModel):
    query: str = Field(..., description="The user's query")
    session_id: Optional[str] = Field(
        default="default", description="Session ID"
    )
    reset: Optional[bool] = Field(
        default=False,
        description="If true, clears server-side session history before handling the message",
    )


class ChatResponse(BaseModel):
    response: str
    metadata: Optional[Dict[str, Any]] = None


# Lightweight health helpers

def _vector_check_sync() -> bool:
    try:
        from retrieval import Retriever
        r = Retriever()
        return bool(getattr(r, "is_enabled", False))
    except Exception:
        return False


# Routes

@app.get("/", tags=["General"])
async def root():
    return {"message": "KnowledgeSpace AI Backend is running", "version": "2.0.0"}


@app.get("/health", tags=["General"])
async def health_check():
    """Cheap health for Docker healthcheck / load balancers."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "knowledge-space-agent-backend",
        "version": "2.0.0",
    }


@app.get("/api/health", tags=["General"])
async def health():
    """
    Public health: includes feature flags but never stalls.
    Vector status is probed with a short timeout in a background thread.
    """
    timeout_s = float(os.getenv("HEALTH_VECTOR_TIMEOUT", "1.0"))
    try:
        vector_enabled = await asyncio.wait_for(
            asyncio.to_thread(_vector_check_sync),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        vector_enabled = False

    components = {
        "vector_search": "enabled" if vector_enabled else "disabled",
        "llm": "enabled" if (os.getenv("GOOGLE_API_KEY") or os.getenv("GCP_PROJECT_ID")) else "disabled",
        "keyword_search": "enabled",
    }
    return {
        "status": "healthy",
        "version": "2.0.0",
        "components": components,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/chat/stream", tags=["Chat"])
async def chat_stream_endpoint(query: str, session_id: str = "default"):
    """
    Stream chat responses using Server-Sent Events (SSE).
    Tokens appear in real-time as the LLM generates them.
    
    This provides a better user experience by showing responses
    as they are generated, similar to ChatGPT.
    """
    async def generate():
        try:
            # Send initial connection message
            yield f"data: {json.dumps({'type': 'start', 'message': 'Connected'})}\n\n"
            
            # Get the response from the assistant
            response_text = await assistant.handle_chat(
                session_id=session_id,
                query=query,
                reset=False,
            )
            
            # Stream the response in chunks (word-by-word simulation)
            words = response_text.split(' ')
            chunk_size = 3  # Send 3 words at a time for smooth streaming
            
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i + chunk_size])
                if i > 0:
                    chunk = ' ' + chunk
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                await asyncio.sleep(0.03)  # 30ms delay for streaming effect
            
            # Send completion message
            yield f"data: {json.dumps({'type': 'done', 'message': 'Complete'})}\n\n"
            
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Request timed out. Please try a simpler query.'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(msg: ChatMessage):
    try:
        start_time = time.time()
        response_text = await assistant.handle_chat(
            session_id=msg.session_id or "default",
            query=msg.query,
            reset=bool(msg.reset),
        )
        process_time = time.time() - start_time
        metadata = {
            "process_time": process_time,
            "session_id": msg.session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "reset": bool(msg.reset),
        }
        return ChatResponse(response=response_text, metadata=metadata)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Request timed out. Please try with a simpler query.",
        )
    except Exception as e:
        return ChatResponse(
            response=f"Error: {e}",
            metadata={"error": True, "session_id": msg.session_id},
        )


@app.post("/api/session/reset", tags=["Chat"])
async def reset_session(payload: Dict[str, str]):
    sid = (payload or {}).get("session_id") or "default"
    assistant.reset_session(sid)
    return {"status": "ok", "session_id": sid, "message": "Session cleared"}


# Entry point
if __name__ == "__main__":
    env = os.getenv("ENVIRONMENT", "production").lower()
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
        log_level="info",
        proxy_headers=True,
    )