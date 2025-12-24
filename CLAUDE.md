# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG (Retrieval-Augmented Generation) application for structural engineering codes. Provides an AI assistant with two bundles:
- **Colombian Bundle** (ES/EN): NSR-10 (Colombian mandatory standard) + ACI-318 (reference code)
- **PBDE Bundle** (EN only): ASCE-7 + LATBSDC + ACI-318 for performance-based design peer review

## Key Commands

```bash
# Run the Streamlit application
streamlit run app.py --server.port 8501

# Process PDFs with Chunkr.ai (requires CHUNKR_API_KEY in .env)
# Note: Also requires chunkr-ai and PyPDF2 packages (not in requirements.txt)
python ingest_chunkr.py

# Build/rebuild ChromaDB from processed chunks (requires OPENAI_API_KEY in .env)
python build_vectordb_chunkr.py

# Fix LaTeX formatting issues in NSR-10 chunks and rebuild database
python fix_nsr10_latex.py
```

## Architecture

### Data Pipeline
1. **PDF Ingestion** (`ingest_chunkr.py`): Uses Chunkr.ai API to parse PDFs (NSR-10, ACI-318, ASCE-7, LATBSDC). Uses LLM (vision model) for tables/formulas/images and standard OCR for text/titles.
2. **Chunk Storage** (`chunkr_output/`): JSON files with chunks containing: `content`, `code`, `pages`, `segment_types`, `article_references`, `has_table`, `has_formula`, `has_image`.
3. **Vector Database** (`build_vectordb_chunkr.py`): Creates ChromaDB with OpenAI `text-embedding-3-small` embeddings. Prefixes chunks with `[CODE]` tag when not already present to improve retrieval.
4. **Application** (`app.py`): Streamlit UI with RAG retrieval and LLM (`gpt-4.1-mini`) response generation.

### Main Application Structure (app.py)

- **Configuration** (~line 54): `MODEL_NAME`, `EMBEDDING_MODEL_NAME`, `PERSIST_DIR`, `COLLECTION_NAME`
- **UI Text Dictionaries**: Bilingual ES/EN interface strings in `UI_TEXT` dict
- **Google Sheets Integration**: Optional user feedback logging via `gspread`
- **LaTeX Formatting**: `convert_latex_delimiters()` converts `\[...\]` to `$$...$$` for Streamlit rendering
- **Retrieval System**:
  - `extract_article_references()`: Parses article citations (e.g., "C.21.12.4.4", "25.4.2.1") from queries
  - `keyword_search_documents()`: Direct text search for specific articles
  - `resolve_references_recursively()`: Follows cross-references to related articles
  - `BalancedCodeRetriever`: Custom LangChain `BaseRetriever` subclass
- **Query Processing**: Prompt templates with instructions for citing sources, response generation with streaming

### Key Components

**BalancedCodeRetriever**: Custom retriever that:
- Combines semantic search (`vectorstore.similarity_search_with_score`) with keyword-based article lookup
- Balances results between NSR-10 and ACI-318 based on user preference (configurable ratio)
- Supports recursive cross-reference expansion (optional feature via `resolve_cross_references` parameter)
- Article reference patterns: NSR-10 uses `C.21.12.4.4` format (letter prefix), ACI-318 uses `25.4.2.1` format (number prefix)

**LaTeX Handling**: Source PDFs contain LaTeX formulas. The app converts `\[...\]` to `$$...$$` for Streamlit rendering. `fix_nsr10_latex.py` wraps orphaned LaTeX commands in `$` delimiters.

## Environment Variables

Required in `.env` or `.streamlit/secrets.toml`:
- `OPENAI_API_KEY`: For embeddings and LLM
- `CHUNKR_API_KEY`: For PDF processing (only needed for ingestion)
- Google Sheets credentials (optional, for feedback logging)

## Data Files

- `chunkr_output/NSR-10_chunks.json`: ~4,558 chunks from NSR-10
- `chunkr_output/ACI-318_chunks.json`: ~2,013 chunks from ACI-318
- `chunkr_output/ASCE-7_chunks.json`: ~1,460 chunks from ASCE-7 (pages 382-889)
- `chunkr_output/LATBSDC_chunks.json`: ~201 chunks from LATBSDC Guidelines
- `chunkr_output/NSR-10_chunks_fixed.json`: LaTeX-fixed version used by database
- `chroma_db_chunkr/`: Persistent ChromaDB with embeddings (~8,232 total documents)

## Dependencies

ChromaDB is pinned to `>=1.3.0,<2.0.0` for database compatibility. If upgrading, you may need to rebuild the database.

## Known Issues

- NSR-10 has OCR quality issues on some pages where formulas appear as garbled text (e.g., `f'Ai` instead of `$\sqrt{f'_c}A_j$`)
- ACI-318 was processed with better quality and renders formulas correctly
- ASCE-7 is partially processed (pages 382-889 only) due to upload timeouts during ingestion. Pages 1-381 are missing and can be re-processed later.
