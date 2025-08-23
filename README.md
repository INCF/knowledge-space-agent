# KnowledgeSpace AI Agent - Dataset Search Assistant

A web application that provides an AI-powered chatbot interface for dataset discovery, using Google Gemini API on the backend and a React-based frontend.

## Prerequisites

- **Python**: 3.11 or higher
- **Google API Key** for Gemini
- **UV package manager** (for backend environment & dependencies)

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd folder name
```

### 2. Install UV package manager

- **Windows**:
  ```bash
  pip install uv
  ```
- **macOS/Linux**:
  Follow the official guide:
  https://docs.astral.sh/uv/getting-started/installation/

### 3. Configure environment variables

Create a file named `.env` in the project root with the following content (without quotes):

```
GOOGLE_API_KEY=your_google_api_key_here
```

> **Note:** Do not commit `.env` to version control.

### 4. Create and activate a virtual environment

```bash
# Create a virtual environment using UV
uv venv

# Activate it:
# On Windows (cmd):
 .venv/bin/activate

```

### 5. Install backend dependencies

With the virtual environment activated:

```bash
uv sync
```

### 6. Install frontend dependencies

```bash
cd frontend
npm install

```

## Running the Application

### Backend (port 8000)

In one terminal, from the project root with the virtual environment active:

```bash
uv run main.py
```

- By default, this will start the backend server on port 8000. Adjust configuration if you need a different port.

### Frontend (port 5000)

In another terminal:

```bash
cd frontend
npm start
```

- This will start the React development server, typically on http://localhost:5000.

## Accessing the Application

Open your browser to:

```
http://localhost:5000
```

The frontend will communicate with the backend at port 8000.

## Running with Docker

To build and start both the backend and frontend in containers:

```
docker-compose up --build
```

**Frontend** →``http://localhost:3000``

**Backend health** → ``http://localhost:8000/api/health``

# Scraping & Data Preprocessing for KnowledgeSpace Datasets

This repository provides a set of Python scripts and modules to ingest, clean, and enrich neuroscience metadata from Google Cloud Storage, as well as scrape  identifiers and references from linked resources.Key features:

- **Elasticsearch Scraping**: The `ksdata_scraping.py` script harvests raw dataset records directly from our Elasticsearch cluster and writes them to GCS. It uses a Point-In-Time (PIT) scroll to page through each index safely, authenticating via credentials stored in your environment.
- **GCS I/O**: Download raw JSON lists from `gs://ks_datasets/raw_dataset/...` and upload preprocessed outputs to `gs://ks_datasets/preprocessed_data/...`.
- **HTML Cleaning**: Strip or convert embedded HTML (e.g. `<a>` tags) into plain text or Markdown.
- **URL Extraction**: Find and dedupe all links in descriptions and metadata for later retrieval.
- **Chunk Construction**: Build semantic “chunks” by concatenating fields (title, description, context labels, etc.) for downstream vectorization.
- **Metadata Filters**: Assemble structured metadata dictionaries (`species`, `region`, `keywords`, `identifier1…n`, etc.) for each record.
- **Per-Datasource Preprocessing**: Each data source has its own preprocessing script (e.g. `scr_017041_dandi.py`, `scr_006274_neuroelectro_ephys.py`) saved in `gcs://ks_datasets/preprocessed_data/`.
- **Extensible Configs**: Easily add new datasources by updating GCS paths and field mappings.

## Additional Notes

- **Environment**: Make sure `.env` is present before starting the backend.
- **Ports**: If ports 5000 or 8000 are in use, adjust scripts/configuration accordingly.
- **UV Commands**:
  - `uv venv` creates the virtual environment.
  - `uv sync` installs dependencies as defined in your project’s config.
- **Troubleshooting**:
  - Verify Python version (`python --version`) and that dependencies installed correctly.
  - Ensure the `.env` file syntax is correct (no extra quotes).
  - For frontend issues, check Node.js version (`node --version`) and logs in terminal.
