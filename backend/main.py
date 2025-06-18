import os
import sys
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from knowledgespace_api import global_search_datasets, format_datasets_list
from openai import OpenAI

# Load environment variables
load_dotenv()

app = FastAPI(title="KnowledgeSpace AI", description="Dataset Discovery Assistant")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
google_api_key = os.getenv("GOOGLE_API_KEY")
PAGE_SIZE = 5
MODEL_NAME = "gemini-2.0-flash"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

class ChatMessage(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    error: bool = False

def gemini(messages: List[Dict[str, Any]], model: str = MODEL_NAME) -> str:
    """Exact implementation as provided by user"""
    try:
        gemini_client = OpenAI(
            api_key=google_api_key,
            base_url=BASE_URL
        )
        resp = gemini_client.chat.completions.create(
            model=model,
            messages=messages  # type: ignore
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

def process_dataset_query(query: str) -> str:
    if not query.strip():
        raise HTTPException(status_code=400, detail="No query entered")

    try:
        # Search KnowledgeSpace datasets
        results = global_search_datasets(query, page=0, per_page=PAGE_SIZE)
        formatted = format_datasets_list(results)

        # Messages for Gemini
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": (
                    f"I searched KnowledgeSpace for datasets matching \"{query}\".\n\n"
                    f"{formatted}\n\n"
                    "Based on the above, summarize what kinds of datasets were found "
                    "and give working links for the more information "
                )
            }
        ]

        # Get Gemini response
        answer = gemini(messages)
        return answer

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "KnowledgeSpace AI Backend", "status": "running"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage):
    """Handle chat messages and return AI responses"""
    try:
        response = process_dataset_query(message.query)
        return ChatResponse(response=response, error=False)
    except HTTPException as e:
        return ChatResponse(response=e.detail, error=True)
    except Exception as e:
        return ChatResponse(response=f"An unexpected error occurred: {str(e)}", error=True)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "KnowledgeSpace AI"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
