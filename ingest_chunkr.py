"""
Chunkr Document Ingestion Script
Processes structural engineering code PDFs for RAG application.

Supported documents:
- NSR-10: Colombian Seismic-Resistant Construction Code
- ACI-318: American Concrete Institute Building Code
- ASCE-7: Minimum Design Loads and Associated Criteria
- LATBSDC: Los Angeles Tall Buildings Structural Design Council Guidelines

Configuration:
- Tables, Formulas, Pictures: Vision model (LLM) for better structure
- Text, Titles, Lists: Standard OCR (Auto) for cost efficiency

Usage:
1. Ensure CHUNKR_API_KEY is set in .env
2. Run: python ingest_chunkr.py
3. Output saved to chunkr_output/
"""

import os
import sys
import json
import time
import asyncio
from datetime import datetime
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Check for required packages
try:
    from chunkr_ai import Chunkr
    from chunkr_ai.models import (
        Configuration,
        ChunkProcessing,
        SegmentProcessing,
        GenerationConfig,
        GenerationStrategy,
    )
except ImportError:
    print("ERROR: chunkr-ai not installed")
    print("Run: pip install chunkr-ai")
    sys.exit(1)

try:
    import PyPDF2
except ImportError:
    print("ERROR: PyPDF2 not installed")
    print("Run: pip install PyPDF2")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

# PDFs to process
# NOTE: Comment/uncomment documents as needed to avoid re-processing
DOCUMENTS = [
    {
        "path": "pdfs/NSR-10.pdf",
        "code": "NSR-10",
        "description": "Colombian Seismic-Resistant Construction Code"
    },
    {
        "path": "pdfs/ACI-318-19.pdf",
        "code": "ACI-318",
        "description": "American Concrete Institute Building Code"
    },
    {
        "path": "pdfs/ASCE7-16.pdf",
        "code": "ASCE-7",
        "description": "ASCE 7-16 Minimum Design Loads and Associated Criteria"
    },
    {
        "path": "pdfs/LATBSDC_Guidelines.pdf",
        "code": "LATBSDC",
        "description": "Los Angeles Tall Buildings Structural Design Council Guidelines"
    },
]

# Output directory
OUTPUT_DIR = "chunkr_output"

# Chunk configuration for RAG
TARGET_CHUNK_LENGTH = 512  # tokens per chunk

# Maximum file size for Chunkr API (in MB)
MAX_FILE_SIZE_MB = 30  # Split files larger than 30MB for reliable uploads


def create_config() -> Configuration:
    """
    Create Chunkr configuration optimized for structural code documents.

    - LLM strategy for: tables, formulas, pictures (complex elements)
    - Auto strategy for: text, titles, lists (standard content)
    """
    return Configuration(
        # Chunk processing for RAG
        chunk_processing=ChunkProcessing(
            target_length=TARGET_CHUNK_LENGTH
        ),

        # Segment-specific processing strategies
        segment_processing=SegmentProcessing(
            # Use LLM (vision model) for complex elements
            table=GenerationConfig(
                strategy=GenerationStrategy.LLM,
                html=GenerationStrategy.LLM,
                markdown=GenerationStrategy.LLM,
            ),
            formula=GenerationConfig(
                strategy=GenerationStrategy.LLM,
                markdown=GenerationStrategy.LLM,
            ),
            picture=GenerationConfig(
                strategy=GenerationStrategy.LLM,
                description=True,  # Generate descriptions for images
            ),

            # Use Auto (standard OCR) for text content - cost efficient
            text=GenerationConfig(
                strategy=GenerationStrategy.AUTO,
            ),
            title=GenerationConfig(
                strategy=GenerationStrategy.AUTO,
            ),
            section_header=GenerationConfig(
                strategy=GenerationStrategy.AUTO,
            ),
            list_item=GenerationConfig(
                strategy=GenerationStrategy.AUTO,
            ),
            caption=GenerationConfig(
                strategy=GenerationStrategy.AUTO,
            ),
            page_header=GenerationConfig(
                strategy=GenerationStrategy.AUTO,
            ),
            page_footer=GenerationConfig(
                strategy=GenerationStrategy.AUTO,
            ),
        ),
    )


def split_pdf(input_path: str, output_dir: str, max_size_mb: float = 90) -> list:
    """
    Split a PDF into smaller parts if it exceeds max size.

    Returns list of paths to split files (or original if no split needed).
    """
    file_size_mb = os.path.getsize(input_path) / (1024 * 1024)

    if file_size_mb <= max_size_mb:
        return [input_path]

    print(f"  File exceeds {max_size_mb}MB limit, splitting...")

    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, 'rb') as infile:
        reader = PyPDF2.PdfReader(infile)
        total_pages = len(reader.pages)

        # Estimate pages per chunk based on file size
        pages_per_chunk = int(total_pages * max_size_mb / file_size_mb * 0.9)  # 10% safety margin
        pages_per_chunk = max(50, pages_per_chunk)  # Minimum 50 pages per chunk

        parts = []
        base_name = Path(input_path).stem

        for start_page in range(0, total_pages, pages_per_chunk):
            end_page = min(start_page + pages_per_chunk, total_pages)
            part_num = len(parts) + 1

            output_path = os.path.join(output_dir, f"{base_name}_part{part_num}.pdf")

            writer = PyPDF2.PdfWriter()
            for i in range(start_page, end_page):
                writer.add_page(reader.pages[i])

            with open(output_path, 'wb') as outfile:
                writer.write(outfile)

            part_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"    Part {part_num}: pages {start_page+1}-{end_page} ({part_size_mb:.1f} MB)")

            parts.append({
                "path": output_path,
                "start_page": start_page,
                "end_page": end_page,
                "part_num": part_num
            })

        return parts


def process_document(api_key: str, doc_info: dict, config: Configuration, temp_dir: str) -> dict:
    """
    Process a single document with Chunkr.

    Args:
        api_key: Chunkr API key
        doc_info: Document metadata (path, code, description)
        config: Chunkr configuration
        temp_dir: Directory for temporary split files

    Returns:
        Processing result with task data and statistics
    """
    pdf_path = doc_info["path"]
    code = doc_info["code"]

    print(f"\n{'=' * 60}")
    print(f"Processing: {code}")
    print(f"File: {pdf_path}")
    print('=' * 60)

    if not os.path.exists(pdf_path):
        print(f"ERROR: File not found: {pdf_path}")
        return None

    # Get file size
    file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")

    # Split if needed
    parts = split_pdf(pdf_path, temp_dir, MAX_FILE_SIZE_MB)

    all_chunks = []
    all_segment_counts = {}
    total_pages = 0
    total_time = 0
    task_ids = []

    for i, part in enumerate(parts):
        part_path = part if isinstance(part, str) else part["path"]
        part_num = 1 if isinstance(part, str) else part["part_num"]

        if len(parts) > 1:
            print(f"\n--- Processing Part {part_num} of {len(parts)} ---")

        start_time = time.time()
        print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
        print("Uploading and processing...")

        try:
            # Create fresh client for each part
            chunkr = Chunkr(api_key=api_key)

            # Upload and wait for completion
            task = chunkr.upload(part_path, config)

            elapsed = time.time() - start_time
            total_time += elapsed

            print(f"  Completed in {elapsed:.1f} seconds")
            print(f"  Task ID: {task.task_id}")
            task_ids.append(task.task_id)

            # Get results
            task_json = task.json()
            output = task_json.get('output')

            if output is None:
                print(f"  WARNING: No output returned for task {task.task_id}")
                chunkr.close()
                continue

            # Get chunks and adjust page numbers if this is a split part
            chunks = output.get('chunks', [])
            page_offset = 0 if isinstance(part, str) else part.get("start_page", 0)

            for chunk in chunks:
                # Adjust page numbers for segments
                for seg in chunk.get('segments', []):
                    if seg.get('page_number'):
                        seg['page_number'] += page_offset
                all_chunks.append(chunk)

                # Count segments
                for seg in chunk.get('segments', []):
                    seg_type = seg.get('segment_type', 'Unknown')
                    all_segment_counts[seg_type] = all_segment_counts.get(seg_type, 0) + 1

            part_pages = output.get('page_count', 0)
            total_pages += part_pages
            print(f"  Pages: {part_pages}, Chunks: {len(chunks)}")

            chunkr.close()

        except Exception as e:
            print(f"ERROR processing part: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_chunks:
        print(f"ERROR: No chunks extracted for {code}")
        return None

    # Build combined result
    print(f"\nCombined Results for {code}:")
    print(f"  Total pages: {total_pages}")
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"  Total segments: {sum(all_segment_counts.values())}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"\n  Segment types:")
    for seg_type, count in sorted(all_segment_counts.items(), key=lambda x: -x[1]):
        print(f"    {seg_type}: {count}")

    # Build combined task_json structure
    combined_json = {
        "task_ids": task_ids,
        "output": {
            "chunks": all_chunks,
            "page_count": total_pages
        }
    }

    return {
        "code": code,
        "task_ids": task_ids,
        "task_json": combined_json,
        "page_count": total_pages,
        "chunk_count": len(all_chunks),
        "segment_counts": all_segment_counts,
        "processing_time": total_time,
    }


def save_results(results: list, output_dir: str):
    """Save processing results to files."""

    os.makedirs(output_dir, exist_ok=True)

    for result in results:
        if result is None:
            continue

        code = result["code"]
        task_json = result["task_json"]

        # Custom JSON encoder for datetime objects
        def json_serial(obj):
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")

        # Save full JSON output
        json_file = os.path.join(output_dir, f"{code}_chunkr_output.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(task_json, f, ensure_ascii=False, indent=2, default=json_serial)
        print(f"Saved: {json_file}")

        # Save chunks in a format ready for ChromaDB ingestion
        chunks_file = os.path.join(output_dir, f"{code}_chunks.json")
        chunks_for_db = []

        output = task_json.get('output', {})
        for chunk in output.get('chunks', []):
            # Extract all segments in this chunk
            segments = chunk.get('segments', [])

            # Get chunk content
            content = chunk.get('content', '') or chunk.get('embed', '')

            # Get page numbers covered by this chunk
            pages = list(set(seg.get('page_number') for seg in segments if seg.get('page_number')))

            # Get segment types in this chunk
            seg_types = list(set(seg.get('segment_type') for seg in segments))

            # Extract article references from content
            import re
            article_refs = []
            patterns = [
                r'[A-Z]\.\d+(?:\.\d+)+',  # NSR-10: C.21.12.4.4
                r'\d+\.\d+(?:\.\d+)+',     # ACI-318: 25.4.2.1
            ]
            for pattern in patterns:
                refs = re.findall(pattern, content)
                article_refs.extend(refs)
            article_refs = list(set(article_refs))

            chunks_for_db.append({
                "chunk_id": chunk.get('chunk_id'),
                "content": content,
                "code": code,
                "pages": sorted(pages) if pages else [],
                "segment_types": seg_types,
                "article_references": article_refs,
                "chunk_length": chunk.get('chunk_length', 0),
                "has_table": 'Table' in seg_types,
                "has_formula": 'Formula' in seg_types,
                "has_image": 'Picture' in seg_types,
            })

        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_for_db, f, ensure_ascii=False, indent=2)
        print(f"Saved: {chunks_file} ({len(chunks_for_db)} chunks)")


def main():
    print("=" * 60)
    print("CHUNKR DOCUMENT INGESTION")
    print("Processing NSR-10 and ACI-318 for RAG Application")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")

    # Check API key
    api_key = os.environ.get("CHUNKR_API_KEY")
    if not api_key:
        print("\nERROR: CHUNKR_API_KEY not set in .env")
        sys.exit(1)

    print(f"\nAPI Key: {api_key[:8]}...{api_key[-4:]}")

    # Create configuration
    config = create_config()
    print(f"\nConfiguration:")
    print(f"  Chunk target length: {TARGET_CHUNK_LENGTH} tokens")
    print(f"  LLM processing: tables, formulas, pictures")
    print(f"  Auto processing: text, titles, lists, headers")
    print(f"  Max file size: {MAX_FILE_SIZE_MB} MB (will split larger files)")

    # Check which documents exist
    print(f"\nDocuments to process:")
    docs_to_process = []
    for doc in DOCUMENTS:
        exists = os.path.exists(doc["path"])
        status = "✓" if exists else "✗ NOT FOUND"
        print(f"  [{status}] {doc['code']}: {doc['path']}")
        if exists:
            docs_to_process.append(doc)

    if not docs_to_process:
        print("\nERROR: No documents found to process")
        sys.exit(1)

    # Confirm before processing
    total_estimate = sum(
        os.path.getsize(doc["path"]) / (1024 * 1024)
        for doc in docs_to_process
    )
    print(f"\nTotal size: {total_estimate:.1f} MB")
    print("\nThis will use Chunkr API credits.")

    response = input("Continue? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        sys.exit(0)

    # Create temp directory for split files
    temp_dir = os.path.join(OUTPUT_DIR, "temp_splits")
    os.makedirs(temp_dir, exist_ok=True)

    # Process documents
    results = []
    for doc in docs_to_process:
        result = process_document(api_key, doc, config, temp_dir)
        results.append(result)

    # Save results
    print(f"\n{'=' * 60}")
    print("SAVING RESULTS")
    print('=' * 60)
    save_results(results, OUTPUT_DIR)

    # Summary
    print(f"\n{'=' * 60}")
    print("PROCESSING COMPLETE")
    print('=' * 60)

    total_pages = 0
    total_chunks = 0
    total_time = 0

    for result in results:
        if result:
            print(f"\n{result['code']}:")
            print(f"  Pages: {result['page_count']}")
            print(f"  Chunks: {result['chunk_count']}")
            print(f"  Time: {result['processing_time']:.1f}s")
            total_pages += result['page_count']
            total_chunks += result['chunk_count']
            total_time += result['processing_time']

    print(f"\nTotals:")
    print(f"  Pages processed: {total_pages}")
    print(f"  Chunks created: {total_chunks}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    print(f"\nOutput saved to: {OUTPUT_DIR}/")
    print(f"  - *_chunkr_output.json (full API response)")
    print(f"  - *_chunks.json (ready for ChromaDB)")

    print(f"\nNext step: Run build_vectordb.py to create ChromaDB")
    print(f"\nFinished: {datetime.now().isoformat()}")

    # Cleanup temp files
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary files.")


if __name__ == "__main__":
    main()
