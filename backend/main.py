# main.py
import os
import time
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import json

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from agents import NeuroscienceAssistant

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limit configuration
RATE_LIMIT = os.getenv("RATE_LIMIT", "10/minute")

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# FastAPI app + CORS
app = FastAPI(
    title="KnowledgeSpace AI",
    description="Neuroscience Dataset Discovery Assistant",
    version="2.0.0",
)

# Attach limiter to app
app.state.limiter = limiter


# Custom rate limit exceeded handler
@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    logger.warning(
        f"Rate limit exceeded for IP: {get_remote_address(request)} "
        f"on path: {request.url.path}"
    )
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Too many requests. Please wait and try again.",
            "retry_after": str(exc.detail),
        },
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


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
@limiter.limit(RATE_LIMIT)
async def chat_endpoint(request: Request, msg: ChatMessage):
    try:
        start_time = time.time()

        # Log the request
        client_ip = get_remote_address(request)
        logger.info(
            f"Chat request from {client_ip} | "
            f"session: {msg.session_id} | "
            f"query length: {len(msg.query)}"
        )

        response_text = await assistant.handle_chat(
            session_id=msg.session_id or "default",
            query=msg.query,
            reset=bool(msg.reset),
        )
        process_time = time.time() - start_time

        logger.info(
            f"Chat response sent to {client_ip} | "
            f"process_time: {process_time:.2f}s"
        )

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
        logger.error(f"Chat error: {e}")
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
    logger.info(f"Starting server with rate limit: {RATE_LIMIT}")
    env = os.getenv("ENVIRONMENT", "production").lower()
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
        log_level="info",
        proxy_headers=True,
    )