import json
import re
import sqlite3
from datetime import datetime

import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import requests

from ingest import extract_text_from_pdf

# ---------- Config ----------
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"
NUM_CTX = 16384

PDF_PATH = "data/raw/infosys_q1_fy25.pdf"
COMPANY = "Infosys"
QUARTER = "Q1 FY25"
DB_PATH = "data/processed/sentiment_history.db"
VECTOR_DB_PATH = "data/vector_db"
COLLECTION_NAME = f"{COMPANY.lower()}_earnings"

VALID_SENTIMENTS = {"Positive", "Neutral", "Negative"}


# ---------- Vector DB Setup ----------
def get_collection():
    """Connect to the existing ChromaDB collection."""
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text"
    )
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=ollama_ef
    )


# ---------- LLM call with schema enforcement ----------
def llm_call(system, user, schema):
    """Call Ollama with grammar-constrained JSON output."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": schema,
        "keep_alive": "10m",
        "options": {
            "num_ctx": NUM_CTX,
            "temperature": 0,
        },
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=900)
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])


# ---------- Schemas ----------
TOPICS_SCHEMA = {
    "type": "object",
    "properties": {
        "topics": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
        }
    },
    "required": ["topics"],
}

DETAIL_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["Positive", "Neutral", "Negative"]},
        "quote": {"type": "string"},
    },
    "required": ["sentiment", "quote"],
}


# ---------- Deterministic verification ----------
def normalize(text):
    """Collapse whitespace and normalize typography so quotes can be matched."""
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_exists(quote, transcript_norm):
    """Check the quote is actually in the source, not hallucinated."""
    q = normalize(quote)
    if len(q) < 20:
        return False
    if q in transcript_norm:
        return True
    return q[:60] in transcript_norm


# ---------- RAG Retrieval ----------
def retrieve_relevant_chunks(collection, topic, n_results=5):
    """Retrieve the most relevant chunks for a given topic."""
    results = collection.query(
        query_texts=[topic],
        n_results=n_results,
        where={
            "$and": [
                {"section": "qa"},
                {"quarter": QUARTER}
            ]
        }
    )
    
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    
    # Format chunks with speaker attribution
    formatted = []
    for chunk, meta, dist in zip(chunks, metadatas, distances):
        formatted.append(f"[{meta['speaker']}]: {chunk}")
    
    return "\n\n".join(formatted), distances

# ---------- Pipeline ----------
def extract_topics(transcript):
    print("\n[1/4] Identifying strategic topics...")
    system = (
        "You are a financial analyst. Identify the 3 most important strategic "
        "topics discussed by management in this earnings call. "
        "Each topic must be a short noun phrase, 2-6 words, with no markdown. "
        "Examples: 'Operating Margin Pressure', 'Large Deal Wins', 'Generative AI Adoption'."
    )
    result = llm_call(system, transcript, TOPICS_SCHEMA)
    topics = [t.strip() for t in result["topics"]]
    for t in topics:
        print(f"   - {t}")
    return topics


def extract_detail_with_rag(collection, topic, transcript_norm):
    print(f"\n[2/4] Extracting quote + sentiment for: {topic}")
    
    # Retrieve relevant chunks
    relevant_context, distances = retrieve_relevant_chunks(collection, topic)
    
    print(f"   Retrieved {len(distances)} chunks (top distance: {distances[0]:.4f})")
    
    system = (
        f"You are a financial analyst reviewing relevant excerpts from an earnings call transcript.\n"
        f"Topic: {topic}\n\n"
        f"Relevant excerpts:\n{relevant_context}\n\n"
        f"1. Copy ONE verbatim sentence from the excerpts above that best reflects "
        f"management's position on this topic. Copy it exactly, character for character.\n"
        f"2. Classify management's sentiment on this topic as Positive, Neutral, or Negative."
    )
    
    result = llm_call(system, "Extract the quote and sentiment.", DETAIL_SCHEMA)

    sentiment = result["sentiment"]
    quote = re.sub(r"\s+", " ", result["quote"]).strip()
    verified = quote_exists(quote, transcript_norm)

    print(f"   sentiment : {sentiment}")
    print(f"   verified  : {'YES' if verified else 'NO  <-- quote not found in source'}")
    print(f"   quote     : {quote[:110]}...")

    return {
        "name": topic,
        "sentiment": sentiment,
        "quote": quote,
        "quote_verified": int(verified),
    }


def analyze_transcript(transcript, collection):
    transcript_norm = normalize(transcript)
    topics = extract_topics(transcript)

    rows = []
    for topic in topics:
        if len(topic) > 80 or "*" in topic or "#" in topic:
            print(f"   REJECTED malformed topic: {topic}")
            continue
        rows.append(extract_detail_with_rag(collection, topic, transcript_norm))

    return rows


# ---------- Storage ----------
def save_rows(rows):
    if not rows:
        print("\nNo valid rows to save.")
        return

    for row in rows:
        row["company"] = COMPANY
        row["quarter"] = QUARTER
        row["analyzed_date"] = datetime.now().strftime("%Y-%m-%d")

    df = pd.DataFrame(rows)
    conn = sqlite3.connect(DB_PATH)

    try:
        conn.execute(
            "DELETE FROM qoq_sentiments WHERE company = ? AND quarter = ?",
            (COMPANY, QUARTER),
        )
    except sqlite3.OperationalError:
        pass

    df.to_sql("qoq_sentiments", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    print(f"\nSaved {len(rows)} rows to {DB_PATH}")


if __name__ == "__main__":
    print(f"Reading {PDF_PATH}...")
    transcript = extract_text_from_pdf(PDF_PATH)
    print(f"Extracted {len(transcript)} characters")
    
    print("\nConnecting to Vector DB...")
    collection = get_collection()
    print(f"✅ Connected to collection '{COLLECTION_NAME}'")
    
    print("\nThis makes 1 full-transcript call (topics) + 3 RAG calls (quotes).")
    print("Expected time: ~1-2 minutes (much faster than before).\n")

    rows = analyze_transcript(transcript, collection)

    print("\n" + "=" * 60)
    print(json.dumps(rows, indent=2))
    print("=" * 60)

    verified = sum(r["quote_verified"] for r in rows)
    print(f"\nQuote verification: {verified}/{len(rows)} quotes found verbatim in source")

    save_rows(rows)