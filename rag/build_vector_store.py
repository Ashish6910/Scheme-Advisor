"""
✅ RAG Pipeline

- Converts scraped schemes → documents
- Cleans + deduplicates data
- Splits into chunks
- Generates embeddings (local)
- Stores in ChromaDB

Run:
    python rag/build_vector_store.py
"""

import json
import os
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# =========================
# ✅ Configuration
# =========================
load_dotenv()

DATA_FILE = "schemes_mygov_data.json"
CHROMA_PATH = "./chroma.db"

# ✅ Embedding model (fast + good)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ✅ Chunking strategy
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80
)

# ===========================
# ✅ Clean + Data validation
# ===========================
def clean_schemes(raw_schemes):
    """Remove bad / duplicate data"""
    unique = {}

    for s in raw_schemes:
        if not s.get("name") or not s.get("url"):
            continue

        if len(s.get("description", "")) < 40:
            continue

        unique[s["url"]] = s  # remove duplicates

    return list(unique.values())


# =================================
# ✅ Schems conversion to document
# =================================
def scheme_to_document(scheme):
    content = f"""
### Scheme Name
{scheme.get('name', '')}

### Category
{scheme.get('category', '')}

### State
{scheme.get('state', '')}

### Ministry
{scheme.get('ministry', '')}

### Description
{scheme.get('description', '')}

### Eligibility
{scheme.get('eligibility', '')}

### Benefits
{scheme.get('benefits', '')}

### Documents Required
{scheme.get('documents', '')}

### How to Apply
{scheme.get('how_to_apply', '')}
""".strip()

    return Document(
        page_content=content,
        metadata={
            "name": scheme.get("name", "")[:200],
            "url": scheme.get("url", ""),
            "category": scheme.get("category", "")[:200],
            "state": scheme.get("state", "")[:100],
            "benefits": scheme.get("benefits", "")[:800],
            "eligibility": scheme.get("eligibility", "")[:800],
        }
    )


# =========================
# ✅ Building vector store
# =========================
def build_vector_store():
    print("=" * 60)
    print("🚀 BUILDING VECTOR STORE")
    print("=" * 60)

    if not os.path.exists(DATA_FILE):
        print("❌ ERROR: schemes_mygov_data.json not found!")
        print("Run scraper first.")
        return

    # ✅ Load data
    with open(DATA_FILE, encoding="utf-8") as f:
        raw_schemes = json.load(f)

    print(f"📊 Raw schemes loaded: {len(raw_schemes)}")

    # ✅ Clean data
    schemes = clean_schemes(raw_schemes)
    print(f"✅ Clean schemes: {len(schemes)}")

    # ✅ Convert to documents
    docs = []
    for scheme in schemes:
        doc = scheme_to_document(scheme)
        chunks = text_splitter.split_documents([doc])
        docs.extend(chunks)

    print(f"📄 Total chunks created: {len(docs)}")

    # ✅ Delete old DB (optional but recommended)
    if os.path.exists(CHROMA_PATH):
        import shutil
        shutil.rmtree(CHROMA_PATH)
        print("♻️ Old vector DB removed")

    print("🧠 Generating embeddings...")

    # ✅ Create vector DB
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    vector_store.persist()

    print(f"\n✅ Stored {vector_store._collection.count()} chunks")
    print("🎯 Ready for chatbot!")
    print("Next: streamlit run ui/chatbot.py")


# =========================
# ✅ Loading Vector Store
# =========================
def get_vector_store():
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )


# ===========================
# ✅ Search schemes function
# ===========================
def search_schemes(query: str, k: int = 10):
    vector_store = get_vector_store()

    results = vector_store.similarity_search(query, k=k)

    output = []

    for doc in results:
        output.append({
            "name": doc.metadata.get("name", ""),
            "url": doc.metadata.get("url", ""),
            "category": doc.metadata.get("category", ""),
            "state": doc.metadata.get("state", ""),
            "benefits": doc.metadata.get("benefits", ""),
            "eligibility": doc.metadata.get("eligibility", ""),
            "content": doc.page_content
        })

    return output


# =========================
# ✅ Main function
# =========================
if __name__ == "__main__":
    build_vector_store()