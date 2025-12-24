"""
Build ChromaDB from Chunkr Output
Creates a vector database from the processed chunks with enriched metadata.

Usage:
    python build_vectordb_chunkr.py
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("ERROR: chromadb not installed")
    print("Run: pip install chromadb")
    sys.exit(1)

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    print("ERROR: langchain-openai not installed")
    print("Run: pip install langchain-openai")
    sys.exit(1)


# Configuration
CHUNKR_OUTPUT_DIR = "chunkr_output"
CHROMA_DB_DIR = "chroma_db_chunkr"
COLLECTION_NAME = "structural_codes_chunkr"

# Batch size for embedding
BATCH_SIZE = 100


def load_chunks() -> list:
    """Load chunks from Chunkr output files."""
    all_chunks = []

    # All supported chunk files (order: Colombian codes, then US codes)
    chunk_files = [
        "NSR-10_chunks.json",
        "ACI-318_chunks.json",
        "ASCE-7_chunks.json",
        "LATBSDC_chunks.json",
    ]

    for filename in chunk_files:
        filepath = os.path.join(CHUNKR_OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
                all_chunks.extend(chunks)
                print(f"  Loaded {len(chunks)} chunks from {filename}")

    return all_chunks


def create_embeddings_model():
    """Create OpenAI embeddings model."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key
    )


def build_database(chunks: list, embeddings_model):
    """Build ChromaDB from chunks."""

    # Initialize ChromaDB
    print(f"\nInitializing ChromaDB at {CHROMA_DB_DIR}...")

    # Remove existing database
    if os.path.exists(CHROMA_DB_DIR):
        import shutil
        shutil.rmtree(CHROMA_DB_DIR)

    client = chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=Settings(anonymized_telemetry=False)
    )

    # Create collection
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Structural codes processed by Chunkr.ai"}
    )

    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Total chunks to process: {len(chunks)}")

    # Process in batches
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(0, len(chunks), BATCH_SIZE):
        batch_num = batch_idx // BATCH_SIZE + 1
        batch_chunks = chunks[batch_idx:batch_idx + BATCH_SIZE]

        print(f"\n  Batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)...")

        # Prepare data
        documents = []
        metadatas = []
        ids = []

        for i, chunk in enumerate(batch_chunks):
            chunk_id = chunk.get('chunk_id') or f"chunk_{batch_idx + i}"

            content = chunk.get('content', '')
            if not content:
                continue

            # Augment content with code identifier for better semantic search
            # This helps when users ask "what does NSR-10 say about X" but the
            # chunk text doesn't explicitly mention the code name
            code = chunk.get('code', 'unknown')
            content_lower = content.lower()

            if code == 'NSR-10' and 'nsr-10' not in content_lower and 'nsr10' not in content_lower:
                content = f"[NSR-10] {content}"
            elif code == 'ACI-318' and 'aci-318' not in content_lower and 'aci318' not in content_lower:
                content = f"[ACI-318] {content}"
            elif code == 'ASCE-7' and 'asce-7' not in content_lower and 'asce7' not in content_lower and 'asce 7' not in content_lower:
                content = f"[ASCE-7] {content}"
            elif code == 'LATBSDC' and 'latbsdc' not in content_lower:
                content = f"[LATBSDC] {content}"

            # Build metadata
            metadata = {
                "code": chunk.get('code', 'unknown'),
                "pages": json.dumps(chunk.get('pages', [])),
                "segment_types": json.dumps(chunk.get('segment_types', [])),
                "article_references": json.dumps(chunk.get('article_references', [])),
                "has_table": chunk.get('has_table', False),
                "has_formula": chunk.get('has_formula', False),
                "has_image": chunk.get('has_image', False),
                "chunk_length": chunk.get('chunk_length', 0),
            }

            # Add first page as integer for filtering
            pages = chunk.get('pages', [])
            if pages:
                metadata["page"] = pages[0]

            documents.append(content)
            metadatas.append(metadata)
            ids.append(chunk_id)

        if not documents:
            continue

        # Generate embeddings
        try:
            embeddings = embeddings_model.embed_documents(documents)

            # Add to collection
            collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            print(f"    Added {len(documents)} documents")

        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    # Final stats
    print(f"\n{'=' * 60}")
    print("DATABASE COMPLETE")
    print('=' * 60)

    final_count = collection.count()
    print(f"  Total documents: {final_count}")
    print(f"  Database location: {CHROMA_DB_DIR}")

    return client


def verify_database():
    """Verify the database works with a sample query."""
    print(f"\n{'=' * 60}")
    print("VERIFICATION")
    print('=' * 60)

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    embeddings_model = create_embeddings_model()

    # Test queries (Colombian and US codes)
    test_queries = [
        "requisitos de refuerzo longitudinal en vigas",
        "development length of reinforcement",
        "C.21.6.4 columnas",
        "seismic design category",  # ASCE-7
        "performance based design acceptance criteria",  # LATBSDC
    ]

    for query in test_queries:
        print(f"\n  Query: '{query}'")

        query_embedding = embeddings_model.embed_query(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )

        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            code = meta.get('code', 'unknown')
            pages = json.loads(meta.get('pages', '[]'))
            print(f"    {i+1}. [{code}] Page {pages[0] if pages else '?'} (dist: {dist:.3f})")
            print(f"       {doc[:100]}...")


def main():
    print("=" * 60)
    print("BUILD CHROMADB FROM CHUNKR OUTPUT")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\nERROR: OPENAI_API_KEY not set in .env")
        sys.exit(1)

    # Load chunks
    print("\nLoading chunks...")
    chunks = load_chunks()

    if not chunks:
        print("ERROR: No chunks found")
        sys.exit(1)

    print(f"  Total chunks: {len(chunks)}")

    # Create embeddings model
    print("\nInitializing embeddings model...")
    embeddings_model = create_embeddings_model()

    # Build database
    build_database(chunks, embeddings_model)

    # Verify
    verify_database()

    print(f"\nFinished: {datetime.now().isoformat()}")
    print("\nNext: Update app.py to use the new database")


if __name__ == "__main__":
    main()
