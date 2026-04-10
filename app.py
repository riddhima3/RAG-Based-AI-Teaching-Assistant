import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import time
import pickle
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="RAG Teaching Assistant", layout="wide")

st.title("RAG-Based AI Teaching Assistant")
st.write("Upload lecture audio or text, process it into a searchable knowledge base, and ask questions.")

st.divider()

# ── Lazy imports with error handling ─────────────────────────────────────────

@st.cache_resource
def load_whisper():
    try:
        import whisper
        model = whisper.load_model("base")
        return model
    except Exception as e:
        return None

@st.cache_resource
def load_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        return None

# ── Helper functions ──────────────────────────────────────────────────────────

def transcribe_audio(file_path):
    model = load_whisper()
    if model is None:
        st.error("Whisper not installed. Run: pip install openai-whisper")
        return ""
    result = model.transcribe(file_path)
    return result["text"]


def clean_text(text):
    import re
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\?\!\:\;\-\(\)]', '', text)
    return text.strip()


def semantic_chunk(text, chunk_size=300, overlap=50):
    """Split text into overlapping chunks with metadata."""
    words = text.split()
    chunks = []
    i = 0
    chunk_id = 0

    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "word_count": len(chunk_words),
            "start_word": i,
            "end_word": i + len(chunk_words)
        })
        chunk_id += 1
        i += chunk_size - overlap

    return chunks


def generate_embeddings(chunks):
    embedder = load_sentence_transformer()
    if embedder is None:
        st.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        return None
    texts = [c["text"] for c in chunks]
    with st.spinner("Generating embeddings..."):
        embeddings = embedder.encode(texts, show_progress_bar=False)
    return embeddings


def cosine_similarity_search(query_embedding, embeddings, top_k=3):
    from numpy.linalg import norm
    sims = []
    for i, emb in enumerate(embeddings):
        sim = np.dot(query_embedding, emb) / (norm(query_embedding) * norm(emb) + 1e-10)
        sims.append((i, float(sim)))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]


def build_prompt(query, retrieved_chunks):
    context = "\n\n".join([f"[Chunk {c['chunk_id']+1}]: {c['text']}" for c in retrieved_chunks])
    prompt = f"""You are a helpful teaching assistant. Use the lecture content below to answer the student's question clearly and accurately.

Lecture Content:
{context}

Student Question: {query}

Answer:"""
    return prompt


def call_llm(prompt, api_key, model="claude-sonnet-4-20250514"):
    """Call Anthropic Claude API."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"Error calling LLM: {str(e)}"


def save_knowledge_base(chunks, embeddings, source_name):
    os.makedirs("knowledge_base", exist_ok=True)
    kb = {
        "source": source_name,
        "chunks": chunks,
        "embeddings": embeddings.tolist()
    }
    path = f"knowledge_base/{source_name.replace(' ', '_')}.json"
    with open(path, "w") as f:
        json.dump(kb, f)
    return path


def load_knowledge_bases():
    kb_dir = Path("knowledge_base")
    if not kb_dir.exists():
        return {}
    kbs = {}
    for f in kb_dir.glob("*.json"):
        with open(f, "r") as fp:
            data = json.load(fp)
            kbs[data["source"]] = data
    return kbs


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "Process Lecture",
    "Knowledge Base",
    "Ask a Question",
    "Pipeline Overview"
])

# ── Tab 1: Process Lecture ────────────────────────────────────────────────────

with tab1:
    st.subheader("Step 1 — Upload Lecture Content")
    st.write("Upload an audio file to transcribe, or paste/upload lecture text directly.")

    input_type = st.radio("Input Type", ["Text / Transcript", "Audio File (MP3/WAV)"], horizontal=True)

    lecture_text = ""
    source_name = ""

    if input_type == "Audio File (MP3/WAV)":
        audio_file = st.file_uploader("Upload Audio", type=["mp3", "wav", "m4a"])
        source_name = st.text_input("Lecture Name", placeholder="e.g. ML Lecture 3 - Gradient Descent")

        if audio_file and source_name:
            if st.button("Transcribe Audio"):
                os.makedirs("temp", exist_ok=True)
                temp_path = f"temp/{audio_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(audio_file.read())

                with st.spinner("Transcribing with Whisper..."):
                    lecture_text = transcribe_audio(temp_path)
                    os.remove(temp_path)

                if lecture_text:
                    st.session_state['raw_text'] = lecture_text
                    st.session_state['source_name'] = source_name
                    st.success("Transcription complete!")
                    with st.expander("View Transcript"):
                        st.write(lecture_text)

    else:
        source_name = st.text_input("Lecture Name", placeholder="e.g. ML Lecture 3 - Gradient Descent")
        text_input_type = st.radio("Provide text via", ["Paste Text", "Upload TXT File"], horizontal=True)

        if text_input_type == "Paste Text":
            lecture_text = st.text_area("Paste Lecture Transcript Here", height=300)
        else:
            txt_file = st.file_uploader("Upload TXT File", type=["txt"])
            if txt_file:
                lecture_text = txt_file.read().decode("utf-8")
                st.success(f"Loaded {len(lecture_text.split())} words")
                with st.expander("Preview"):
                    st.write(lecture_text[:1000] + "...")

    st.divider()
    st.subheader("Step 2 — Configure Chunking")

    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.slider("Chunk Size (words)", 100, 500, 300, 50,
                               help="Number of words per chunk")
    with col2:
        overlap = st.slider("Overlap (words)", 0, 100, 50, 10,
                            help="Words shared between consecutive chunks")

    st.divider()
    st.subheader("Step 3 — Process and Embed")

    if st.button("Process Lecture", use_container_width=True):
        text = lecture_text or st.session_state.get('raw_text', '')
        name = source_name or st.session_state.get('source_name', '')

        if not text.strip():
            st.warning("Please provide lecture text or transcribe an audio file first.")
        elif not name.strip():
            st.warning("Please enter a lecture name.")
        else:
            with st.spinner("Cleaning text..."):
                cleaned = clean_text(text)
                time.sleep(0.3)

            with st.spinner("Chunking text..."):
                chunks = semantic_chunk(cleaned, chunk_size, overlap)
                time.sleep(0.3)

            st.info(f"Created {len(chunks)} chunks from {len(cleaned.split())} words")

            embeddings = generate_embeddings(chunks)

            if embeddings is not None:
                path = save_knowledge_base(chunks, embeddings, name)
                st.session_state['last_processed'] = name
                st.success(f"Knowledge base saved — {len(chunks)} chunks, {embeddings.shape[1]}-dim embeddings")

                # Preview chunks
                st.subheader("Chunk Preview")
                df_chunks = pd.DataFrame([{
                    "Chunk": c["chunk_id"] + 1,
                    "Words": c["word_count"],
                    "Preview": c["text"][:100] + "..."
                } for c in chunks])
                st.dataframe(df_chunks, use_container_width=True)

# ── Tab 2: Knowledge Base ─────────────────────────────────────────────────────

with tab2:
    st.subheader("Saved Knowledge Bases")

    kbs = load_knowledge_bases()

    if not kbs:
        st.info("No knowledge bases yet. Process a lecture in the first tab.")
    else:
        for name, kb in kbs.items():
            with st.expander(f"📚 {name}"):
                chunks = kb["chunks"]
                embeddings = np.array(kb["embeddings"])

                c1, c2, c3 = st.columns(3)
                c1.metric("Chunks", len(chunks))
                c2.metric("Embedding Dimensions", embeddings.shape[1])
                c3.metric("Total Words", sum(c["word_count"] for c in chunks))

                df = pd.DataFrame([{
                    "Chunk": c["chunk_id"] + 1,
                    "Words": c["word_count"],
                    "Text Preview": c["text"][:120] + "..."
                } for c in chunks])
                st.dataframe(df, use_container_width=True)

# ── Tab 3: Ask a Question ─────────────────────────────────────────────────────

with tab3:
    st.subheader("Ask a Question")

    kbs = load_knowledge_bases()

    if not kbs:
        st.warning("No knowledge bases found. Please process a lecture first.")
    else:
        selected_kb = st.selectbox("Select Lecture Knowledge Base", list(kbs.keys()))

        st.divider()

        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Your Question", placeholder="e.g. What is gradient descent?")
        with col2:
            top_k = st.number_input("Top K Chunks", min_value=1, max_value=5, value=3)

        use_llm = st.checkbox("Generate AI Answer (requires Anthropic API key)")

        api_key = ""
        if use_llm:
            api_key = st.text_input("Anthropic API Key", type="password",
                                    help="Get your key from console.anthropic.com")

        if st.button("Search", use_container_width=True):
            if not query.strip():
                st.warning("Please enter a question.")
            else:
                kb = kbs[selected_kb]
                chunks = kb["chunks"]
                embeddings = np.array(kb["embeddings"])

                embedder = load_sentence_transformer()
                if embedder is None:
                    st.error("sentence-transformers not available.")
                else:
                    with st.spinner("Searching knowledge base..."):
                        query_embedding = embedder.encode([query])[0]
                        results = cosine_similarity_search(query_embedding, embeddings, top_k)

                    retrieved = [chunks[i] for i, _ in results]
                    scores = [s for _, s in results]

                    st.divider()
                    st.subheader("Retrieved Chunks")

                    for idx, (chunk, score) in enumerate(zip(retrieved, scores)):
                        with st.expander(f"Chunk {chunk['chunk_id']+1} — Similarity: {round(score*100, 1)}%"):
                            st.progress(score)
                            st.write(chunk["text"])

                    if use_llm and api_key:
                        st.divider()
                        st.subheader("AI Answer")
                        with st.spinner("Generating answer..."):
                            prompt = build_prompt(query, retrieved)
                            answer = call_llm(prompt, api_key)
                        st.success(answer)

                        with st.expander("View Full Prompt Sent to LLM"):
                            st.code(prompt)

                    elif use_llm and not api_key:
                        st.warning("Please enter your Anthropic API key.")

# ── Tab 4: Pipeline Overview ──────────────────────────────────────────────────

with tab4:
    st.subheader("End-to-End RAG Pipeline")
    st.write("This project implements a complete Retrieval-Augmented Generation pipeline for lecture content.")

    st.divider()

    steps = [
        ("1. Audio Transcription", "Whisper",
         "Converts lecture audio (MP3/WAV) into raw text transcripts using OpenAI Whisper."),
        ("2. Text Preprocessing", "Regex + Clean",
         "Removes noise, normalises whitespace, and prepares text for chunking."),
        ("3. Semantic Chunking", "Custom Python",
         "Splits transcript into overlapping word-based chunks with metadata (chunk ID, word count, position)."),
        ("4. Embedding Generation", "Sentence Transformers",
         "Converts each chunk into a 384-dim dense vector using all-MiniLM-L6-v2."),
        ("5. Vector Storage", "JSON / NumPy",
         "Stores embeddings and chunk metadata in a local JSON knowledge base."),
        ("6. Similarity Retrieval", "Cosine Similarity",
         "At query time, encodes the question and retrieves the top-K most similar chunks."),
        ("7. Prompt Engineering", "Custom Template",
         "Constructs a context-aware prompt from retrieved chunks and the student's question."),
        ("8. LLM Response", "Anthropic Claude",
         "Sends the prompt to Claude to generate a grounded, accurate answer."),
    ]

    for title, tech, desc in steps:
        with st.expander(f"{title} — `{tech}`"):
            st.write(desc)

    st.divider()
    st.subheader("Data Flow")
    st.code("""
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
    """)
