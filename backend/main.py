import os
import time
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import agent orchestrator
from agents import NeuroscienceAssistant

# Configuration

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

print("Initializing  Assistant...")
assistant = NeuroscienceAssistant()
print("Assistant initialized successfully!")


# Request/Response Models

class ChatMessage(BaseModel):
    query: str = Field(..., description="The user's query about neuroscience datasets")
    session_id: Optional[str] = Field(default="default", description="Session ID for conversation context")

class ChatResponse(BaseModel):
    response: str
    metadata: Optional[Dict[str, Any]] = None
    
    
# Middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# API Routes

@app.get("/", tags=["General"])
async def root():
    return {"message": "KnowledgeSpace AI Backend is running", "version": "2.0.0"}

@app.get("/api/health", tags=["General"])
async def health():
    """Health check endpoint."""
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
    """
    Main chat endpoint for querying neuroscience datasets.
    """
    try:
        print(f"[{datetime.utcnow().isoformat()}] Query: {msg.query[:100]}... (session: {msg.session_id})")
        
        start_time = time.time()
        response_text = await assistant.handle_chat(
            session_id=msg.session_id or "default",
            query=msg.query
        )
        process_time = time.time() - start_time
        
        metadata = {
            "process_time": process_time,
            "session_id": msg.session_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return ChatResponse(response=response_text, metadata=metadata)
        
    except asyncio.TimeoutError:
        print(f"Timeout processing query: {msg.query}")
        raise HTTPException(
            status_code=504,
            detail="Request timed out. Please try with a simpler query."
        )
    except Exception as e:
        print(f"Error processing query: {e}")
        import traceback
        traceback.print_exc()
        
        return ChatResponse(
            response=(
                "I encountered an error while searching for datasets. "
                "Please try rephrasing your query.\n\n"
                "Tips for better results:\n"
                "• Be specific about brain regions or techniques\n"
                "• Include the species if relevant\n"
                "• Use standard neuroscience terminology\n\n"
                "Example: 'medial prefrontal cortex recordings in rats'"
            ),
            metadata={"error": True, "session_id": msg.session_id}
        )

@app.get("/api/stats", tags=["Admin"])
async def get_stats():
    """Get usage statistics."""
    return {
        "active_sessions": len(assistant.chat_history),
        "total_queries": sum(len(h) // 2 for h in assistant.chat_history.values()),
        "timestamp": datetime.utcnow().isoformat()
    }

# Error Handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "path": str(request.url.path)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": "Please try again later"}
    )

# Startup
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("ENVIRONMENT", "development") == "development"
    
    print(f"Starting server on {host}:{port}")
    print(f"Reload mode: {reload}")
    print(f"API docs available at: http://{host}:{port}/docs")
    
    uvicorn.run("main:app", host=host, port=port, reload=reload, log_level="info")
