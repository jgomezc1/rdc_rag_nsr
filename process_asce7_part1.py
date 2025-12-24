"""
Process ASCE-7 Pages 1-381 (missing from initial ingestion)

This script:
1. Extracts pages 1-381 from ASCE7-16.pdf into smaller chunks
2. Processes each chunk with Chunkr.ai
3. Merges results with existing ASCE-7_chunks.json
4. Rebuilds the vector database

Usage:
    python process_asce7_part1.py
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import PyPDF2
except ImportError:
    print("ERROR: PyPDF2 not installed")
    print("Run: pip install PyPDF2")
    sys.exit(1)

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


# Configuration
INPUT_PDF = "pdfs/ASCE7-16.pdf"
OUTPUT_DIR = "chunkr_output"
TEMP_DIR = "chunkr_output/temp_asce7_parts"
TARGET_CHUNK_LENGTH = 512
PAGES_PER_PART = 100  # Small parts to avoid timeout


def create_config() -> Configuration:
    """Create Chunkr configuration."""
    return Configuration(
        chunk_processing=ChunkProcessing(target_length=TARGET_CHUNK_LENGTH),
        segment_processing=SegmentProcessing(
            table=GenerationConfig(strategy=GenerationStrategy.LLM, html=GenerationStrategy.LLM, markdown=GenerationStrategy.LLM),
            formula=GenerationConfig(strategy=GenerationStrategy.LLM, markdown=GenerationStrategy.LLM),
            picture=GenerationConfig(strategy=GenerationStrategy.LLM, description=True),
            text=GenerationConfig(strategy=GenerationStrategy.AUTO),
            title=GenerationConfig(strategy=GenerationStrategy.AUTO),
            section_header=GenerationConfig(strategy=GenerationStrategy.AUTO),
            list_item=GenerationConfig(strategy=GenerationStrategy.AUTO),
            caption=GenerationConfig(strategy=GenerationStrategy.AUTO),
            page_header=GenerationConfig(strategy=GenerationStrategy.AUTO),
            page_footer=GenerationConfig(strategy=GenerationStrategy.AUTO),
        ),
    )


def split_pdf_pages(input_path: str, output_dir: str, start_page: int, end_page: int, pages_per_part: int) -> list:
    """Split specific page range into smaller parts."""
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, 'rb') as infile:
        reader = PyPDF2.PdfReader(infile)
        total_pages = len(reader.pages)

        if end_page > total_pages:
            end_page = total_pages

        parts = []
        for start in range(start_page - 1, end_page, pages_per_part):
            end = min(start + pages_per_part, end_page)
            part_num = len(parts) + 1

            output_path = os.path.join(output_dir, f"ASCE7_part1_{part_num}.pdf")

            writer = PyPDF2.PdfWriter()
            for i in range(start, end):
                writer.add_page(reader.pages[i])

            with open(output_path, 'wb') as outfile:
                writer.write(outfile)

            part_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  Part {part_num}: pages {start+1}-{end} ({part_size_mb:.1f} MB)")

            parts.append({
                "path": output_path,
                "start_page": start,
                "end_page": end,
                "part_num": part_num
            })

        return parts


def process_parts(api_key: str, parts: list, config: Configuration) -> list:
    """Process PDF parts with Chunkr."""
    all_chunks = []

    for part in parts:
        print(f"\n--- Processing Part {part['part_num']} (pages {part['start_page']+1}-{part['end_page']}) ---")
        print(f"Started: {datetime.now().strftime('%H:%M:%S')}")

        start_time = time.time()

        try:
            chunkr = Chunkr(api_key=api_key)
            task = chunkr.upload(part["path"], config)

            elapsed = time.time() - start_time
            print(f"  Completed in {elapsed:.1f} seconds")
            print(f"  Task ID: {task.task_id}")

            task_json = task.json()
            output = task_json.get('output')

            if output is None:
                print(f"  WARNING: No output returned")
                chunkr.close()
                continue

            chunks = output.get('chunks', [])
            page_offset = part["start_page"]

            for chunk in chunks:
                for seg in chunk.get('segments', []):
                    if seg.get('page_number'):
                        seg['page_number'] += page_offset
                all_chunks.append(chunk)

            print(f"  Chunks: {len(chunks)}")
            chunkr.close()

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    return all_chunks


def convert_to_db_format(chunks: list) -> list:
    """Convert Chunkr chunks to database format."""
    import re

    chunks_for_db = []

    for chunk in chunks:
        segments = chunk.get('segments', [])
        content = chunk.get('content', '') or chunk.get('embed', '')
        pages = list(set(seg.get('page_number') for seg in segments if seg.get('page_number')))
        seg_types = list(set(seg.get('segment_type') for seg in segments))

        # Extract article references
        article_refs = []
        patterns = [r'\d+\.\d+(?:\.\d+)+']
        for pattern in patterns:
            refs = re.findall(pattern, content)
            article_refs.extend(refs)
        article_refs = list(set(article_refs))

        chunks_for_db.append({
            "chunk_id": chunk.get('chunk_id'),
            "content": content,
            "code": "ASCE-7",
            "pages": sorted(pages) if pages else [],
            "segment_types": seg_types,
            "article_references": article_refs,
            "chunk_length": chunk.get('chunk_length', 0),
            "has_table": 'Table' in str(seg_types),
            "has_formula": 'Formula' in str(seg_types),
            "has_image": 'Picture' in str(seg_types),
        })

    return chunks_for_db


def merge_chunks(existing_file: str, new_chunks: list) -> list:
    """Merge new chunks with existing ones."""
    existing_chunks = []

    if os.path.exists(existing_file):
        with open(existing_file, 'r', encoding='utf-8') as f:
            existing_chunks = json.load(f)
        print(f"  Loaded {len(existing_chunks)} existing chunks")

    # Merge: new chunks (pages 1-381) + existing chunks (pages 382+)
    merged = new_chunks + existing_chunks
    print(f"  Merged total: {len(merged)} chunks")

    return merged


def main():
    print("=" * 60)
    print("PROCESS ASCE-7 PAGES 1-381")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")

    # Check API key
    api_key = os.environ.get("CHUNKR_API_KEY")
    if not api_key:
        print("\nERROR: CHUNKR_API_KEY not set in .env")
        sys.exit(1)

    # Check input file
    if not os.path.exists(INPUT_PDF):
        print(f"\nERROR: {INPUT_PDF} not found")
        sys.exit(1)

    # Step 1: Split PDF
    print(f"\n{'=' * 60}")
    print("STEP 1: SPLIT PDF PAGES 1-381")
    print('=' * 60)

    parts = split_pdf_pages(INPUT_PDF, TEMP_DIR, 1, 381, PAGES_PER_PART)
    print(f"\nCreated {len(parts)} parts")

    # Confirm
    response = input("\nThis will use Chunkr API credits. Continue? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        sys.exit(0)

    # Step 2: Process parts
    print(f"\n{'=' * 60}")
    print("STEP 2: PROCESS WITH CHUNKR")
    print('=' * 60)

    config = create_config()
    raw_chunks = process_parts(api_key, parts, config)

    if not raw_chunks:
        print("\nERROR: No chunks extracted")
        sys.exit(1)

    print(f"\nTotal raw chunks: {len(raw_chunks)}")

    # Step 3: Convert to DB format
    print(f"\n{'=' * 60}")
    print("STEP 3: CONVERT TO DB FORMAT")
    print('=' * 60)

    new_chunks = convert_to_db_format(raw_chunks)
    print(f"  Converted {len(new_chunks)} chunks")

    # Step 4: Merge with existing
    print(f"\n{'=' * 60}")
    print("STEP 4: MERGE WITH EXISTING CHUNKS")
    print('=' * 60)

    existing_file = os.path.join(OUTPUT_DIR, "ASCE-7_chunks.json")
    merged_chunks = merge_chunks(existing_file, new_chunks)

    # Step 5: Save
    print(f"\n{'=' * 60}")
    print("STEP 5: SAVE MERGED CHUNKS")
    print('=' * 60)

    with open(existing_file, 'w', encoding='utf-8') as f:
        json.dump(merged_chunks, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {existing_file}")

    # Cleanup temp files
    import shutil
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        print("  Cleaned up temp files")

    print(f"\n{'=' * 60}")
    print("COMPLETE!")
    print('=' * 60)
    print(f"Total ASCE-7 chunks: {len(merged_chunks)}")
    print("\nNext step: Run 'python build_vectordb_chunkr.py' to rebuild the database")
    print(f"\nFinished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
