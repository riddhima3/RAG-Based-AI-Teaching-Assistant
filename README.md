# RAG-Based AI Teaching Assistant

An end-to-end Retrieval-Augmented Generation (RAG) pipeline that turns lecture
audio or text into an interactive Q&A assistant. Upload a lecture, process it
into a searchable vector knowledge base, and ask questions to get context-aware
answers powered by an LLM.

## How It Works

1. **Transcription** — Upload lecture audio and Whisper converts it to text
2. **Preprocessing** — Text is cleaned and normalised
3. **Chunking** — Transcript is split into overlapping chunks with metadata
4. **Embedding** — Each chunk is converted to a vector using Sentence Transformers
5. **Storage** — Embeddings and metadata are saved as a local knowledge base
6. **Retrieval** — At query time, cosine similarity finds the most relevant chunks
7. **Prompt Engineering** — Retrieved chunks are structured into a context-aware prompt
8. **LLM Response** — Claude generates a grounded answer from the retrieved context

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| OpenAI Whisper | Audio transcription |
| Sentence Transformers | Embedding generation (all-MiniLM-L6-v2) |
| NumPy | Cosine similarity search |
| Pandas | Chunk metadata management |
| Anthropic Claude | LLM response generation |
| Streamlit | Interactive dashboard |

## Features

- 4-tab dashboard — Process Lecture, Knowledge Base, Ask a Question, Pipeline Overview
- Supports audio (MP3/WAV) and text (TXT/paste) input
- Configurable chunk size and overlap from UI
- Local JSON vector store — no external database needed
- Top-K similarity retrieval with similarity scores
- Optional LLM integration with Anthropic Claude API
- Full prompt visible for transparency
- Pipeline architecture diagram built in

## Project Structure

```
rag-teaching-assistant/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
├── LICENSE                 # MIT License
├── knowledge_base/         # Auto-created — stores processed lectures
│   └── *.json              # Chunks + embeddings per lecture
└── temp/                   # Auto-created — temporary audio files
```

## Setup & Installation

1. Clone the repository

```bash
git clone https://github.com/riddhima3/rag-teaching-assistant.git
cd rag-teaching-assistant
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. For audio transcription, also install ffmpeg

```bash
# Mac
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

4. Run the app

```bash
streamlit run app.py
```

## How to Use

1. Go to **Process Lecture** — upload audio or paste text, set chunk size, click Process
2. Go to **Knowledge Base** — view all processed lectures and their chunk details
3. Go to **Ask a Question** — select a lecture, type your question, get retrieved chunks
4. Optionally add your Anthropic API key to get a full AI-generated answer
5. Go to **Pipeline Overview** — see the full data flow architecture

## RAG Pipeline Data Flow

```
Audio/Text Input
      │
      ▼
Whisper Transcription (if audio)
      │
      ▼
Text Cleaning & Preprocessing
      │
      ▼
Semantic Chunking (with overlap)
      │
      ▼
Embedding Generation (Sentence Transformers)
      │
      ▼
Vector Store (local JSON)
      │
   [Query]
      │
      ▼
Query Embedding → Cosine Similarity Search
      │
      ▼
Top-K Chunk Retrieval
      │
      ▼
Prompt Construction (context + question)
      │
      ▼
LLM (Claude) → Context-Aware Answer
```

## Author

**Riddhima Saha**
- LinkedIn: [riddhima-saha](https://www.linkedin.com/in/riddhima-saha)
- Email: riddhima.sahaa@gmail.com
