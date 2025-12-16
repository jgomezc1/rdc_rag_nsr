"""
Chunkr.ai Proof-of-Concept Script
Tests document parsing quality for NSR-10/ACI-318 RAG application.

Usage:
1. Get API key from https://www.chunkr.ai/ (200 free pages)
2. Set environment variable: export CHUNKR_API_KEY="your_key"
3. Run: python test_chunkr_poc.py

This script will:
- Extract a sample of pages from NSR-10 PDF
- Send to Chunkr API for processing
- Display and save the structured output for quality evaluation
"""

import os
import sys
import json
import re
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use system env vars

# Check for required packages
try:
    from chunkr_ai import Chunkr
    from chunkr_ai.models import Configuration, ChunkProcessing
except ImportError:
    print("=" * 60)
    print("MISSING DEPENDENCY: chunkr-ai")
    print("=" * 60)
    print("\nPlease install the Chunkr SDK:")
    print("  pip install chunkr-ai")
    print("\nThen run this script again.")
    sys.exit(1)

try:
    import PyPDF2
except ImportError:
    print("=" * 60)
    print("MISSING DEPENDENCY: PyPDF2")
    print("=" * 60)
    print("\nPlease install PyPDF2:")
    print("  pip install PyPDF2")
    print("\nThen run this script again.")
    sys.exit(1)


def extract_sample_pages(input_pdf: str, output_pdf: str, start_page: int, num_pages: int):
    """
    Extract a sample of pages from a PDF for testing.

    Args:
        input_pdf: Path to the full PDF
        output_pdf: Path for the sample PDF
        start_page: Starting page (0-indexed)
        num_pages: Number of pages to extract
    """
    print(f"\nExtracting {num_pages} pages starting from page {start_page + 1}...")

    with open(input_pdf, 'rb') as infile:
        reader = PyPDF2.PdfReader(infile)
        writer = PyPDF2.PdfWriter()

        total_pages = len(reader.pages)
        end_page = min(start_page + num_pages, total_pages)

        for i in range(start_page, end_page):
            writer.add_page(reader.pages[i])

        with open(output_pdf, 'wb') as outfile:
            writer.write(outfile)

    print(f"  Created: {output_pdf}")
    print(f"  Pages: {start_page + 1} to {end_page} (of {total_pages} total)")
    return output_pdf


def analyze_segments(segments: list) -> dict:
    """
    Analyze the segments returned by Chunkr to assess quality.
    """
    analysis = {
        "total_segments": len(segments),
        "segment_types": {},
        "pages_covered": set(),
        "article_references_found": [],
        "tables_found": 0,
        "formulas_found": 0,
        "avg_content_length": 0,
    }

    total_content_length = 0

    # Patterns for NSR-10 article references
    article_patterns = [
        r'[A-Z]\.\d+(?:\.\d+)+',  # NSR-10: C.21.12.4.4, A.3.2.1
        r'\d+\.\d+(?:\.\d+)+',     # ACI-318: 25.4.2.1
    ]

    for seg in segments:
        # Count segment types
        seg_type = seg.get('segment_type', 'Unknown')
        analysis["segment_types"][seg_type] = analysis["segment_types"].get(seg_type, 0) + 1

        # Track pages
        page_num = seg.get('page_number')
        if page_num:
            analysis["pages_covered"].add(page_num)

        # Get content
        content = seg.get('content', '') or seg.get('markdown', '') or ''
        total_content_length += len(content)

        # Find article references
        for pattern in article_patterns:
            refs = re.findall(pattern, content)
            analysis["article_references_found"].extend(refs)

        # Count tables and formulas
        if seg_type in ['Table', 'TableCell', 'TableRow']:
            analysis["tables_found"] += 1
        if seg_type in ['Formula', 'Equation']:
            analysis["formulas_found"] += 1

    # Calculate averages
    if segments:
        analysis["avg_content_length"] = total_content_length / len(segments)

    # Deduplicate references
    analysis["article_references_found"] = list(set(analysis["article_references_found"]))
    analysis["pages_covered"] = sorted(list(analysis["pages_covered"]))

    return analysis


def display_sample_segments(segments: list, num_samples: int = 5):
    """Display sample segments for quality review."""
    print(f"\n{'=' * 60}")
    print(f"SAMPLE SEGMENTS (showing {min(num_samples, len(segments))} of {len(segments)})")
    print('=' * 60)

    for i, seg in enumerate(segments[:num_samples]):
        print(f"\n--- Segment {i + 1} ---")
        print(f"  Type: {seg.get('segment_type', 'N/A')}")
        print(f"  Page: {seg.get('page_number', 'N/A')}")

        bbox = seg.get('bbox', {})
        if bbox:
            print(f"  BBox: L={bbox.get('left', 0):.1f}, T={bbox.get('top', 0):.1f}, "
                  f"W={bbox.get('width', 0):.1f}, H={bbox.get('height', 0):.1f}")

        content = seg.get('content', '') or seg.get('markdown', '') or ''
        if len(content) > 300:
            content = content[:300] + "..."
        print(f"  Content: {content}")


def main():
    print("=" * 60)
    print("CHUNKR.AI PROOF-OF-CONCEPT TEST")
    print("Testing document parsing quality for NSR-10 RAG application")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")

    # Check for API key
    api_key = os.environ.get("CHUNKR_API_KEY")
    if not api_key:
        print("\n" + "!" * 60)
        print("ERROR: CHUNKR_API_KEY environment variable not set")
        print("!" * 60)
        print("\nTo get an API key:")
        print("  1. Go to https://www.chunkr.ai/")
        print("  2. Sign up (200 free pages included)")
        print("  3. Copy your API key")
        print("  4. Set it: export CHUNKR_API_KEY='your_key_here'")
        print("\nThen run this script again.")
        sys.exit(1)

    print(f"\nAPI Key: {api_key[:8]}...{api_key[-4:]}")

    # Configuration
    SOURCE_PDF = "pdfs/NSR-10.pdf"
    SAMPLE_PDF = "pdfs/NSR-10_sample.pdf"
    OUTPUT_DIR = "chunkr_output"

    # Sample pages that likely contain article references and tables
    # Pages around chapter C (concrete) which has many cross-references
    SAMPLE_START_PAGE = 700  # Approximate location of Título C
    SAMPLE_NUM_PAGES = 10    # 10 pages for POC (within free tier)

    # Verify source PDF exists
    if not os.path.exists(SOURCE_PDF):
        print(f"\nERROR: Source PDF not found: {SOURCE_PDF}")
        sys.exit(1)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Extract sample pages
    print("\n" + "-" * 60)
    print("STEP 1: Extracting sample pages")
    print("-" * 60)

    try:
        extract_sample_pages(SOURCE_PDF, SAMPLE_PDF, SAMPLE_START_PAGE, SAMPLE_NUM_PAGES)
    except Exception as e:
        print(f"ERROR extracting pages: {e}")
        sys.exit(1)

    # Step 2: Process with Chunkr
    print("\n" + "-" * 60)
    print("STEP 2: Processing with Chunkr API")
    print("-" * 60)

    try:
        chunkr = Chunkr(api_key=api_key)

        # Configure chunking for RAG use case
        config = Configuration(
            chunk_processing=ChunkProcessing(target_length=512)
        )

        print(f"\nUploading {SAMPLE_PDF} to Chunkr...")
        print("  (This may take 1-2 minutes)")

        task = chunkr.upload(SAMPLE_PDF, config)

        print(f"\n  Task ID: {task.task_id}")
        print(f"  Status: Completed")

    except Exception as e:
        import traceback
        print(f"\nERROR processing with Chunkr: {e}")
        print(f"\nFull traceback:")
        traceback.print_exc()
        print("\nPossible causes:")
        print("  - Invalid API key")
        print("  - Network connectivity issues")
        print("  - API rate limits exceeded")
        chunkr.close()
        sys.exit(1)

    # Step 3: Extract and analyze output
    print("\n" + "-" * 60)
    print("STEP 3: Analyzing Chunkr output")
    print("-" * 60)

    try:
        # Get full JSON output
        task_json = task.json()

        # Custom JSON encoder for datetime objects
        def json_serial(obj):
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")

        # Save complete output for inspection
        output_file = os.path.join(OUTPUT_DIR, "chunkr_full_output.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(task_json, f, ensure_ascii=False, indent=2, default=json_serial)
        print(f"\nFull output saved to: {output_file}")

        # Get text content
        text_content = task.content()
        text_file = os.path.join(OUTPUT_DIR, "chunkr_text_output.txt")
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text_content)
        print(f"Text content saved to: {text_file}")

        # Extract segments for analysis
        output_data = task_json.get('output', {})
        segments = output_data.get('segments', [])

        print(f"\nTotal segments extracted: {len(segments)}")

    except Exception as e:
        print(f"ERROR extracting output: {e}")
        chunkr.close()
        sys.exit(1)

    # Step 4: Quality analysis
    print("\n" + "-" * 60)
    print("STEP 4: Quality analysis for RAG use case")
    print("-" * 60)

    analysis = analyze_segments(segments)

    print(f"\n=== SEGMENT TYPE DISTRIBUTION ===")
    for seg_type, count in sorted(analysis["segment_types"].items(), key=lambda x: -x[1]):
        print(f"  {seg_type}: {count}")

    print(f"\n=== ARTICLE REFERENCES DETECTED ===")
    print(f"  Total unique references: {len(analysis['article_references_found'])}")
    if analysis['article_references_found']:
        print(f"  Examples: {', '.join(analysis['article_references_found'][:10])}")

    print(f"\n=== CONTENT METRICS ===")
    print(f"  Pages covered: {analysis['pages_covered']}")
    print(f"  Tables found: {analysis['tables_found']}")
    print(f"  Formulas found: {analysis['formulas_found']}")
    print(f"  Avg segment length: {analysis['avg_content_length']:.1f} chars")

    # Display sample segments
    display_sample_segments(segments, num_samples=5)

    # Save analysis
    analysis_file = os.path.join(OUTPUT_DIR, "chunkr_analysis.json")
    # Convert set to list for JSON serialization
    analysis_serializable = {
        **analysis,
        "pages_covered": list(analysis["pages_covered"])
    }
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_serializable, f, ensure_ascii=False, indent=2)
    print(f"\n\nAnalysis saved to: {analysis_file}")

    # Cleanup
    chunkr.close()

    # Step 5: Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS FOR YOUR RAG APPLICATION")
    print("=" * 60)

    print("""
Based on the output, consider:

1. ARTICLE EXTRACTION:
   - If many article references (C.X.X.X) were detected, Chunkr can
     help pre-extract these for your reference expansion feature.

2. TABLE HANDLING:
   - If tables were properly segmented, you can create table-specific
     chunks with preserved structure.

3. CHUNK SIZE:
   - Current config uses 512 tokens. Adjust based on your LLM context.
   - Larger chunks = more context, fewer retrievals
   - Smaller chunks = more precise retrieval, more fragments

4. METADATA ENRICHMENT:
   - Segment types can be stored as metadata
   - Page numbers and bounding boxes available
   - Could extract chapter/section headers for hierarchy

5. NEXT STEPS:
   - Review the JSON output in chunkr_output/
   - Test with different page ranges (try pages with tables/formulas)
   - Estimate full processing cost: ~2,162 pages for NSR-10
""")

    print(f"\nCompleted: {datetime.now().isoformat()}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
