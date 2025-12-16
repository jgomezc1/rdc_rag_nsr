# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG (Retrieval-Augmented Generation) application for Colombian structural engineering codes. Provides an AI assistant that answers questions about NSR-10 (Colombian mandatory standard) and ACI-318 (reference code) using semantic search over processed PDF documents.

## Key Commands

```bash
# Run the Streamlit application
streamlit run app.py --server.port 8501

# Process PDFs with Chunkr.ai (requires CHUNKR_API_KEY in .env)
python ingest_chunkr.py

# Build/rebuild ChromaDB from processed chunks (requires OPENAI_API_KEY in .env)
python build_vectordb_chunkr.py

# Fix LaTeX formatting issues in NSR-10 chunks and rebuild database
python fix_nsr10_latex.py
```

## Architecture

### Data Pipeline
1. **PDF Ingestion** (`ingest_chunkr.py`): Uses Chunkr.ai API to parse NSR-10 and ACI-318 PDFs with vision model for tables/formulas/images
2. **Chunk Storage** (`chunkr_output/`): JSON files containing parsed chunks with metadata (pages, segment types, article references)
3. **Vector Database** (`build_vectordb_chunkr.py`): Creates ChromaDB with OpenAI embeddings from chunks
4. **Application** (`app.py`): Streamlit UI with RAG retrieval and LLM response generation

### Main Application Structure (app.py ~1850 lines)

- **Configuration** (lines 51-58): Model names, database paths, collection name
- **UI Text Dictionaries** (lines 59-178): Bilingual ES/EN interface strings
- **Google Sheets Integration** (lines 179-314): User feedback logging
- **LaTeX Formatting** (lines 795-908): `convert_latex_delimiters()` and `normalize_latex_output()` for math rendering
- **Retrieval System** (lines 911-1130):
  - `extract_article_references()`: Parses article citations from queries
  - `keyword_search_documents()`: Direct text search for specific articles
  - `resolve_references_recursively()`: Follows cross-references to related articles
  - `BalancedCodeRetriever`: Custom LangChain retriever balancing NSR-10/ACI-318 sources
- **Query Processing** (lines 1234-1520): LLM setup, prompt templates, response generation
- **Main UI** (lines 1523-1853): Streamlit layout, chat interface, source display

### Key Components

**BalancedCodeRetriever** (line 1133): Custom retriever that:
- Combines semantic search with keyword-based article lookup
- Balances results between NSR-10 and ACI-318 based on user preference
- Supports recursive cross-reference expansion (optional feature)

**LaTeX Handling**: Source PDFs contain LaTeX formulas. The app converts `\[...\]` to `$$...$$` for Streamlit rendering. Some NSR-10 pages have OCR issues where formulas weren't properly extracted.

## Environment Variables

Required in `.env` or `.streamlit/secrets.toml`:
- `OPENAI_API_KEY`: For embeddings and LLM
- `CHUNKR_API_KEY`: For PDF processing (only needed for ingestion)
- Google Sheets credentials (optional, for feedback logging)

## Data Files

- `chunkr_output/NSR-10_chunks.json`: ~4,558 chunks from NSR-10
- `chunkr_output/ACI-318_chunks.json`: ~2,013 chunks from ACI-318
- `chroma_db_chunkr/`: Persistent ChromaDB with embeddings

## Known Issues

- NSR-10 has OCR quality issues on some pages where formulas appear as garbled text (e.g., `f'Ai` instead of `$\sqrt{f'_c}A_j$`)
- ACI-318 was processed with better quality and renders formulas correctly
