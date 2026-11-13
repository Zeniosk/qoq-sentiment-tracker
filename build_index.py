import chromadb
from chromadb.utils import embedding_functions

from chunking import chunk_transcript
from ingest import extract_text_from_pdf

# --- Configuration ---
PDF_PATH = "data/raw/infosys_q1_fy25.pdf"
COMPANY = "Infosys"
QUARTER = "Q1 FY25"
DB_PATH = "data/vector_db"
COLLECTION_NAME = f"{COMPANY.lower()}_earnings"

def build_index():
    print("📖 Step 1: Reading and chunking transcript...")
    raw_text = extract_text_from_pdf(PDF_PATH)
    chunks = chunk_transcript(raw_text)
    print(f"✅ Generated {len(chunks)} chunks.\n")

    print("🧠 Step 2: Initializing ChromaDB and Ollama Embeddings...")
    client = chromadb.PersistentClient(path=DB_PATH)
    
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text"
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ollama_ef,
        metadata={"description": f"Earnings call transcripts for {COMPANY}"}
    )
    print(f"✅ Collection '{COLLECTION_NAME}' ready.\n")

    print("🗄️ Step 3: Adding chunks to Vector DB (this may take 1-2 mins)...")
    
    documents = [c["text"] for c in chunks]
    ids = [f"{COMPANY.lower()}_{QUARTER.lower().replace(' ', '_')}_chunk_{c['chunk_id']}" for c in chunks]
    
    metadatas = [
        {
            "company": COMPANY,
            "quarter": QUARTER,
            "speaker": c["speaker"],
            "section": c["section"],
            "chunk_id": c["chunk_id"]
        } 
        for c in chunks
    ]

    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    
    print(f"✅ Successfully indexed {len(documents)} chunks into ChromaDB at {DB_PATH}\n")

    print("🔍 Step 4: Running a quick semantic search test...")
    query = "Generative AI impact on large deals"
    
    results = collection.query(
        query_texts=[query],
        n_results=3,
        where={"section": "qa"}
    )
    
    print(f"\nQuery: '{query}' (filtered to Q&A section)")
    print("-" * 60)
    for i, (doc, meta, dist) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["distances"][0]), 1):
        print(f"\n{i}. [{meta['speaker']}] (Distance: {dist:.4f})")
        print(f"   {doc[:180]}...")
    print("-" * 60)

if __name__ == "__main__":
    build_index()