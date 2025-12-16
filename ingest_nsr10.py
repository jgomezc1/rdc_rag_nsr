"""
NSR-10 Ingestion Script
Processes NSR-10 in smaller batches to avoid timeout issues.
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
    sys.exit(1)

try:
    import PyPDF2
except ImportError:
    print("ERROR: PyPDF2 not installed")
    sys.exit(1)

# Configuration
SOURCE_PDF = "pdfs/NSR-10.pdf"
OUTPUT_DIR = "chunkr_output"
TEMP_DIR = os.path.join(OUTPUT_DIR, "nsr10_splits")
PAGES_PER_BATCH = 300  # Process 300 pages at a time
TARGET_CHUNK_LENGTH = 512


def create_config() -> Configuration:
    return Configuration(
        chunk_processing=ChunkProcessing(target_length=TARGET_CHUNK_LENGTH),
        segment_processing=SegmentProcessing(
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
                description=True,
            ),
            text=GenerationConfig(strategy=GenerationStrategy.AUTO),
            title=GenerationConfig(strategy=GenerationStrategy.AUTO),
            section_header=GenerationConfig(strategy=GenerationStrategy.AUTO),
            list_item=GenerationConfig(strategy=GenerationStrategy.AUTO),
            caption=GenerationConfig(strategy=GenerationStrategy.AUTO),
            page_header=GenerationConfig(strategy=GenerationStrategy.AUTO),
            page_footer=GenerationConfig(strategy=GenerationStrategy.AUTO),
        ),
    )


def split_pdf_by_pages(input_pdf: str, output_dir: str, pages_per_batch: int) -> list:
    """Split PDF into batches of pages."""
    os.makedirs(output_dir, exist_ok=True)

    with open(input_pdf, 'rb') as infile:
        reader = PyPDF2.PdfReader(infile)
        total_pages = len(reader.pages)

        print(f"Total pages: {total_pages}")
        print(f"Splitting into batches of {pages_per_batch} pages...")

        parts = []
        base_name = Path(input_pdf).stem

        for start_page in range(0, total_pages, pages_per_batch):
            end_page = min(start_page + pages_per_batch, total_pages)
            part_num = len(parts) + 1

            output_path = os.path.join(output_dir, f"{base_name}_batch{part_num}.pdf")

            writer = PyPDF2.PdfWriter()
            for i in range(start_page, end_page):
                writer.add_page(reader.pages[i])

            with open(output_path, 'wb') as outfile:
                writer.write(outfile)

            part_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  Batch {part_num}: pages {start_page+1}-{end_page} ({part_size_mb:.1f} MB)")

            parts.append({
                "path": output_path,
                "start_page": start_page,
                "end_page": end_page,
                "part_num": part_num
            })

        return parts


def process_batch(api_key: str, batch: dict, config: Configuration, max_retries: int = 3) -> dict:
    """Process a single batch with retries."""
    part_path = batch["path"]
    part_num = batch["part_num"]

    for attempt in range(max_retries):
        try:
            print(f"\n  Attempt {attempt + 1}/{max_retries}...")

            chunkr = Chunkr(api_key=api_key)
            start_time = time.time()

            task = chunkr.upload(part_path, config)

            elapsed = time.time() - start_time
            print(f"    Completed in {elapsed:.1f}s - Task ID: {task.task_id}")

            task_json = task.json()
            output = task_json.get('output')

            chunkr.close()

            if output is None:
                print(f"    WARNING: No output, retrying...")
                continue

            return {
                "task_id": task.task_id,
                "output": output,
                "elapsed": elapsed
            }

        except Exception as e:
            print(f"    ERROR: {e}")
            if attempt < max_retries - 1:
                print(f"    Waiting 30s before retry...")
                time.sleep(30)
            continue

    return None


def main():
    print("=" * 60)
    print("NSR-10 INGESTION")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")

    api_key = os.environ.get("CHUNKR_API_KEY")
    if not api_key:
        print("ERROR: CHUNKR_API_KEY not set")
        sys.exit(1)

    if not os.path.exists(SOURCE_PDF):
        print(f"ERROR: {SOURCE_PDF} not found")
        sys.exit(1)

    config = create_config()

    # Split PDF
    print(f"\nSplitting {SOURCE_PDF}...")
    batches = split_pdf_by_pages(SOURCE_PDF, TEMP_DIR, PAGES_PER_BATCH)
    print(f"Created {len(batches)} batches")

    # Process each batch
    all_chunks = []
    all_segment_counts = {}
    total_pages = 0
    total_time = 0
    task_ids = []

    for batch in batches:
        print(f"\n--- Batch {batch['part_num']} of {len(batches)} (pages {batch['start_page']+1}-{batch['end_page']}) ---")

        result = process_batch(api_key, batch, config)

        if result is None:
            print(f"  FAILED after all retries, skipping batch")
            continue

        task_ids.append(result["task_id"])
        total_time += result["elapsed"]

        output = result["output"]
        chunks = output.get('chunks', [])
        page_offset = batch["start_page"]

        # Adjust page numbers and collect chunks
        for chunk in chunks:
            for seg in chunk.get('segments', []):
                if seg.get('page_number'):
                    seg['page_number'] += page_offset
                seg_type = seg.get('segment_type', 'Unknown')
                all_segment_counts[seg_type] = all_segment_counts.get(seg_type, 0) + 1
            all_chunks.append(chunk)

        batch_pages = output.get('page_count', 0)
        total_pages += batch_pages
        print(f"  Pages: {batch_pages}, Chunks: {len(chunks)}")

    if not all_chunks:
        print("ERROR: No chunks extracted")
        sys.exit(1)

    # Save results
    print(f"\n{'=' * 60}")
    print("SAVING RESULTS")
    print('=' * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Combined JSON
    combined_json = {
        "task_ids": task_ids,
        "output": {
            "chunks": all_chunks,
            "page_count": total_pages
        }
    }

    def json_serial(obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    json_file = os.path.join(OUTPUT_DIR, "NSR-10_chunkr_output.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(combined_json, f, ensure_ascii=False, indent=2, default=json_serial)
    print(f"Saved: {json_file}")

    # Chunks for DB
    import re
    chunks_for_db = []
    for chunk in all_chunks:
        segments = chunk.get('segments', [])
        content = chunk.get('content', '') or chunk.get('embed', '')
        pages = list(set(seg.get('page_number') for seg in segments if seg.get('page_number')))
        seg_types = list(set(seg.get('segment_type') for seg in segments))

        article_refs = []
        for pattern in [r'[A-Z]\.\d+(?:\.\d+)+', r'\d+\.\d+(?:\.\d+)+']:
            refs = re.findall(pattern, content)
            article_refs.extend(refs)

        chunks_for_db.append({
            "chunk_id": chunk.get('chunk_id'),
            "content": content,
            "code": "NSR-10",
            "pages": sorted(pages) if pages else [],
            "segment_types": seg_types,
            "article_references": list(set(article_refs)),
            "chunk_length": chunk.get('chunk_length', 0),
            "has_table": 'Table' in seg_types,
            "has_formula": 'Formula' in seg_types,
            "has_image": 'Picture' in seg_types,
        })

    chunks_file = os.path.join(OUTPUT_DIR, "NSR-10_chunks.json")
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(chunks_for_db, f, ensure_ascii=False, indent=2)
    print(f"Saved: {chunks_file} ({len(chunks_for_db)} chunks)")

    # Summary
    print(f"\n{'=' * 60}")
    print("COMPLETE")
    print('=' * 60)
    print(f"  Pages: {total_pages}")
    print(f"  Chunks: {len(all_chunks)}")
    print(f"  Segments: {sum(all_segment_counts.values())}")
    print(f"  Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\n  Segment types:")
    for seg_type, count in sorted(all_segment_counts.items(), key=lambda x: -x[1]):
        print(f"    {seg_type}: {count}")

    # Cleanup
    import shutil
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        print(f"\nCleaned up temp files.")

    print(f"\nFinished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
