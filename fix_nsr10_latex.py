"""
Fix NSR-10 LaTeX Issues and Rebuild Database

This script:
1. Loads NSR-10 chunks from Chunkr output
2. Fixes orphaned LaTeX (commands not inside $ delimiters)
3. Fixes broken/unclosed delimiters
4. Saves fixed chunks
5. Rebuilds the ChromaDB database

Usage:
    python fix_nsr10_latex.py
"""

import os
import sys
import json
import re
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuration
CHUNKR_OUTPUT_DIR = "chunkr_output"
CHROMA_DB_DIR = "chroma_db_chunkr"
COLLECTION_NAME = "structural_codes_chunkr"
BATCH_SIZE = 100


def fix_latex_in_text(text: str) -> str:
    """
    Fix LaTeX issues in text:
    1. Wrap orphaned LaTeX commands in $ delimiters
    2. Fix unclosed delimiters
    3. Normalize delimiter style
    """

    # Step 1: Convert \[...\] to $$...$$ and \(...\) to $...$
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)

    # Step 2: Protect already-delimited sections
    protected = {}
    counter = [0]

    def protect(match):
        key = f"__PROTECTED_{counter[0]}__"
        counter[0] += 1
        protected[key] = match.group(0)
        return key

    # Protect $$...$$ first (greedy but not across too many lines)
    text = re.sub(r'\$\$[^$]+?\$\$', protect, text)
    # Protect $...$ (single line)
    text = re.sub(r'\$[^$\n]+?\$', protect, text)

    # Step 3: Find and wrap orphaned LaTeX expressions

    # Pattern for complete LaTeX expressions that should be wrapped
    latex_patterns = [
        # Fractions: \frac{...}{...}
        r'\\frac\s*\{[^}]*\}\s*\{[^}]*\}',
        # Square roots: \sqrt{...} or \sqrt[n]{...}
        r'\\sqrt\s*(?:\[[^\]]*\])?\s*\{[^}]*\}',
        # Greek letters with optional subscripts/superscripts
        r'\\(?:alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|Alpha|Beta|Gamma|Delta|Epsilon|Zeta|Eta|Theta|Iota|Kappa|Lambda|Mu|Nu|Xi|Omicron|Pi|Rho|Sigma|Tau|Upsilon|Phi|Chi|Psi|Omega)(?:_\{[^}]*\}|_[a-zA-Z0-9]|\^[a-zA-Z0-9]|\^\{[^}]*\})*',
        # Comparison operators in expressions: X \leq Y, X \geq Y
        r'[A-Za-z0-9_]+\s*\\(?:leq|geq|neq|le|ge|approx|equiv|sim)\s*[A-Za-z0-9_]+',
        # Sum, integral, product with limits
        r'\\(?:sum|int|prod|oint)\s*(?:_\{[^}]*\})?(?:\^\{[^}]*\})?',
    ]

    for pattern in latex_patterns:
        def wrap_if_not_protected(match):
            content = match.group(0)
            # Check if already inside a protected block (shouldn't happen but safety check)
            if '__PROTECTED_' in content:
                return content
            return f'${content}$'

        text = re.sub(pattern, wrap_if_not_protected, text)

    # Step 4: Find isolated LaTeX commands and wrap them with surrounding context
    # This handles cases like: V_n = \frac{A_v f_y d}{s}

    # Pattern for equations containing LaTeX commands
    # Match from start of potential equation to end
    equation_pattern = r'([A-Za-z][A-Za-z0-9_\']*(?:_\{[^}]*\}|_[a-zA-Z0-9])?)\s*=\s*([^.\n]*\\[a-zA-Z]+[^.\n]*?)(?=[.\n]|$)'

    def wrap_equation(match):
        lhs = match.group(1)
        rhs = match.group(2).strip()
        # Only wrap if there's actual LaTeX in the RHS
        if '\\' in rhs:
            return f'${lhs} = {rhs}$'
        return match.group(0)

    text = re.sub(equation_pattern, wrap_equation, text)

    # Step 5: Wrap any remaining isolated backslash commands
    # Pattern: LaTeX command that's clearly mathematical
    isolated_commands = [
        r'\\frac\s*\{[^}]*\}\s*\{[^}]*\}',
        r'\\sqrt\s*\{[^}]*\}',
        r'\\(?:phi|psi|lambda|alpha|beta|gamma|delta|rho|sigma|theta|omega|mu|pi|epsilon)\b',
        r'\\(?:leq|geq|neq|times|cdot|pm|mp|div)\b',
    ]

    for pattern in isolated_commands:
        def wrap_isolated(match):
            content = match.group(0)
            if '__PROTECTED_' in content:
                return content
            return f'${content}$'
        text = re.sub(f'(?<![\\$])({pattern})(?![\\$])', wrap_isolated, text)

    # Step 6: Restore protected sections
    for key, value in protected.items():
        text = text.replace(key, value)

    # Step 7: Clean up any double-wrapped sections ($$$$)
    text = re.sub(r'\${3,}', '$$', text)

    # Step 8: Fix broken delimiters like $...$ $ or $ $...$
    text = re.sub(r'\$\s+\$', '$', text)  # Remove empty $ $
    text = re.sub(r'\$([^$]+)\$\s*\$([^$]+)\$', r'$\1 \2$', text)  # Merge adjacent

    return text


def process_chunks(input_file: str, output_file: str) -> dict:
    """Process chunks from input file and save to output file."""

    print(f"\nProcessing: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    stats = {
        'total': len(chunks),
        'modified': 0,
        'latex_added': 0,
    }

    fixed_chunks = []
    for chunk in chunks:
        original_content = chunk.get('content', '')
        fixed_content = fix_latex_in_text(original_content)

        if fixed_content != original_content:
            stats['modified'] += 1
            # Count new $ signs added
            original_dollars = original_content.count('$')
            fixed_dollars = fixed_content.count('$')
            if fixed_dollars > original_dollars:
                stats['latex_added'] += (fixed_dollars - original_dollars) // 2

        chunk['content'] = fixed_content
        fixed_chunks.append(chunk)

    # Save fixed chunks
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(fixed_chunks, f, ensure_ascii=False, indent=2)

    print(f"  Total chunks: {stats['total']}")
    print(f"  Modified chunks: {stats['modified']}")
    print(f"  LaTeX expressions wrapped: ~{stats['latex_added']}")

    return stats


def rebuild_database():
    """Rebuild ChromaDB from fixed chunks."""

    try:
        import chromadb
        from chromadb.config import Settings
        from langchain_openai import OpenAIEmbeddings
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("REBUILDING CHROMADB")
    print('=' * 60)

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    # Load all chunks
    all_chunks = []

    for filename in ["NSR-10_chunks_fixed.json", "ACI-318_chunks.json"]:
        filepath = os.path.join(CHUNKR_OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
                all_chunks.extend(chunks)
                print(f"  Loaded {len(chunks)} chunks from {filename}")
        else:
            # Fallback to original file for ACI
            original = filename.replace('_fixed', '')
            filepath = os.path.join(CHUNKR_OUTPUT_DIR, original)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                    all_chunks.extend(chunks)
                    print(f"  Loaded {len(chunks)} chunks from {original}")

    print(f"\n  Total chunks: {len(all_chunks)}")

    # Initialize embeddings
    print("\nInitializing embeddings model...")
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key
    )

    # Remove existing database
    if os.path.exists(CHROMA_DB_DIR):
        import shutil
        print(f"  Removing existing database: {CHROMA_DB_DIR}")
        shutil.rmtree(CHROMA_DB_DIR)

    # Initialize ChromaDB
    print(f"  Creating new database: {CHROMA_DB_DIR}")
    client = chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=Settings(anonymized_telemetry=False)
    )

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Structural codes with fixed LaTeX"}
    )

    # Process in batches
    total_batches = (len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(0, len(all_chunks), BATCH_SIZE):
        batch_num = batch_idx // BATCH_SIZE + 1
        batch_chunks = all_chunks[batch_idx:batch_idx + BATCH_SIZE]

        print(f"\n  Batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)...")

        documents = []
        metadatas = []
        ids = []

        for i, chunk in enumerate(batch_chunks):
            chunk_id = chunk.get('chunk_id') or f"chunk_{batch_idx + i}"
            content = chunk.get('content', '')

            if not content:
                continue

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

            pages = chunk.get('pages', [])
            if pages:
                metadata["page"] = pages[0]

            documents.append(content)
            metadatas.append(metadata)
            ids.append(chunk_id)

        if not documents:
            continue

        try:
            embeddings = embeddings_model.embed_documents(documents)

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
    print("DATABASE REBUILD COMPLETE")
    print('=' * 60)
    print(f"  Total documents: {collection.count()}")
    print(f"  Database location: {CHROMA_DB_DIR}")


def verify_fixes():
    """Verify the fixes worked by checking sample chunks."""

    print(f"\n{'=' * 60}")
    print("VERIFICATION")
    print('=' * 60)

    fixed_file = os.path.join(CHUNKR_OUTPUT_DIR, "NSR-10_chunks_fixed.json")

    if not os.path.exists(fixed_file):
        print("  Fixed file not found")
        return

    with open(fixed_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    # Find chunks with V_n formulas
    vn_chunks = [c for c in chunks if 'V_n' in c.get('content', '') and '$' in c.get('content', '')]

    print(f"\n  Chunks with V_n and $ delimiters: {len(vn_chunks)}")

    if vn_chunks:
        print(f"\n  Sample fixed chunk (Page {vn_chunks[0].get('pages', ['?'])[0]}):")
        content = vn_chunks[0].get('content', '')[:800]
        print(f"  {'-' * 50}")
        print(content)


def main():
    print("=" * 60)
    print("FIX NSR-10 LATEX AND REBUILD DATABASE")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")

    # Step 1: Fix NSR-10 chunks
    print(f"\n{'=' * 60}")
    print("STEP 1: FIX NSR-10 LATEX")
    print('=' * 60)

    nsr_input = os.path.join(CHUNKR_OUTPUT_DIR, "NSR-10_chunks.json")
    nsr_output = os.path.join(CHUNKR_OUTPUT_DIR, "NSR-10_chunks_fixed.json")

    if not os.path.exists(nsr_input):
        print(f"ERROR: {nsr_input} not found")
        sys.exit(1)

    process_chunks(nsr_input, nsr_output)

    # Step 2: Verify fixes
    verify_fixes()

    # Step 3: Rebuild database
    rebuild_database()

    print(f"\n{'=' * 60}")
    print("ALL DONE!")
    print('=' * 60)
    print(f"Finished: {datetime.now().isoformat()}")
    print("\nNext steps:")
    print("  1. Restart the Streamlit app")
    print("  2. Test with: ¿Cómo se calcula la resistencia nominal al cortante Vn?")


if __name__ == "__main__":
    main()
