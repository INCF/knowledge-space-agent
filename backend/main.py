import os
import time
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from agents import NeuroscienceAssistant

load_dotenv()

app = FastAPI(
    title="KnowledgeSpace AI",
    description="Neuroscience Dataset Discovery Assistant",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

assistant = NeuroscienceAssistant()


class ChatMessage(BaseModel):
    query: str = Field(..., description="The user's query")
    session_id: Optional[str] = Field(default="default", description="Session ID")
    reset: Optional[bool] = Field(default=False, description="If true, clears server-side session history before handling the message")


class ChatResponse(BaseModel):
    response: str
    metadata: Optional[Dict[str, Any]] = None


@app.get("/", tags=["General"])
async def root():
    return {"message": "KnowledgeSpace AI Backend is running", "version": "2.0.0"}


@app.get("/api/health", tags=["General"])
async def health():
    try:
        from retrieval import Retriever
        retriever = Retriever()
        vector_enabled = retriever.is_enabled
    except Exception:
        vector_enabled = False
    components = {
        "vector_search": "enabled" if vector_enabled else "disabled",
        "llm": "enabled" if os.getenv("GOOGLE_API_KEY") else "disabled",
        "keyword_search": "enabled"
    }
    return {
        "status": "healthy",
        "version": "2.0.0",
        "components": components,
        "timestamp": datetime.utcnow().isoformat()
    }


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
        raise HTTPException(status_code=504, detail="Request timed out. Please try with a simpler query.")
    except Exception as e:
        return ChatResponse(response=f"Error: {e}", metadata={"error": True, "session_id": msg.session_id})


@app.post("/api/session/reset", tags=["Chat"])
async def reset_session(payload: Dict[str, str]):
    sid = payload.get("session_id") or "default"
    assistant.reset_session(sid)
    return {"status": "ok", "session_id": sid, "message": "Session cleared"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("ENVIRONMENT", "development") == "development",
        log_level="info",
    )
