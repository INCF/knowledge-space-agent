# KnowledgeSpace Agent - Examples

This directory contains minimal, reproducible examples to help new contributors and GSoC students understand how the KnowledgeSpace AI Agent works.

## Prerequisites

- **Python**: 3.11 or higher
- **Google API Key**: Required for Gemini LLM (get one free at [Google AI Studio](https://aistudio.google.com/apikey))

## Quick Setup

### 1. Install Dependencies

From the project root:

```bash
# Using UV (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file in the project root (or set environment variables):

```bash
# Required: Your Google Gemini API key
GOOGLE_API_KEY=your_api_key_here

# Use standard Gemini API (not Vertex AI)
GEMINI_USE_VERTEX=false
```

> **Note:** The examples require only `GOOGLE_API_KEY` to run. Other environment variables (BigQuery, Vertex AI Vector Search) are optional and only needed for full production functionality.

## Running the Examples

### Basic Demo Script

```bash
# From the project root
cd examples
python basic_demo.py
```

### What to Expect

The script will:

1. **Initialize** the NeuroscienceAssistant agent
2. **Send** a sample query about neuroscience datasets
3. **Display**:
   - The final synthesized response produced by the agent
   - (Optionally) selected intermediate signals for learning and debugging purposes

**Example Output (illustrative):**

```
============================================================
KnowledgeSpace Agent - Basic Demo
============================================================

Initializing the NeuroscienceAssistant...
✓ Agent initialized successfully

Sending query: "Find datasets about hippocampus neurons in mice"

Processing... (this may take a few seconds)

--- Agent Response ---
### 🔬 Neuroscience Datasets Found

#### 1. Mouse Hippocampus CA1 Recordings
- **Source:** DANDI Archive
- **Description:** Extracellular recordings from hippocampal neurons...
...
```

## Files in This Directory

| File | Purpose |
|------|---------|
| `README.md` | This setup guide |
| `basic_demo.py` | Minimal Python script demonstrating agent usage |
| `local_knowledge.json` | Sample mock dataset entries (for reference only) |

> **Note:** `local_knowledge.json` is provided as illustrative sample data for contributors; the current agent uses remote KnowledgeSpace APIs and does not directly load this file.

## Understanding the Agent Workflow

A simplified view of the agent's high-level workflow:

> This diagram is a conceptual overview intended for learning and onboarding purposes.

```
User Query
    ↓
┌─────────────────────┐
│  1. Extract Keywords │  ← Gemini extracts search terms
│  2. Detect Intents   │  ← Classify query type (data discovery, etc.)
└─────────────────────┘
    ↓
┌─────────────────────┐
│  3. Execute Search   │  ← Query KnowledgeSpace API + Vector DB
└─────────────────────┘
    ↓
┌─────────────────────┐
│  4. Fuse Results     │  ← Combine and rank results
└─────────────────────┘
    ↓
┌─────────────────────┐
│  5. Synthesize       │  ← Gemini generates natural language response
└─────────────────────┘
    ↓
Final Response
```

## Troubleshooting

### "GOOGLE_API_KEY must be set"

Make sure you've set the environment variable:

```bash
# Windows PowerShell
$env:GOOGLE_API_KEY = "your_key_here"

# Windows CMD
set GOOGLE_API_KEY=your_key_here

# Linux/macOS
export GOOGLE_API_KEY=your_key_here
```

### Import errors

Make sure you're running from the correct directory and dependencies are installed:

```bash
cd knowledge-space-agent
uv sync  # or pip install -e .
cd examples
python basic_demo.py
```

### Rate limiting

If you see rate limit errors, wait a few seconds and try again. The free Gemini API tier has request limits.

## Next Steps

- Explore `backend/agents.py` to understand the full agent implementation
- Check `backend/ks_search_tool.py` for KnowledgeSpace API integration
- Visit the [hosted demo](https://chat.knowledge-space.org/) to see the full application

