"""
Extract all tables from NSR-10 and ACI-318 PDFs.
Stores tables in a structured JSON format for querying.
"""
import pdfplumber
import json
import os
import re
from datetime import datetime

def clean_text(text):
    """Clean extracted text."""
    if text is None:
        return ""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', str(text).strip())
    return text

def extract_table_title(page, table_bbox, page_text):
    """Try to extract a title for the table from nearby text."""
    # Look for common table title patterns in the page text
    patterns = [
        r'(Tabla\s+[\w\d\.\-]+(?:\s*[–—-]\s*[^\.]+)?)',
        r'(Table\s+[\w\d\.\-]+(?:\s*[–—-]\s*[^\.]+)?)',
        r'(TABLA\s+[\w\d\.\-]+(?:\s*[–—-]\s*[^\.]+)?)',
        r'(TABLE\s+[\w\d\.\-]+(?:\s*[–—-]\s*[^\.]+)?)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, page_text, re.IGNORECASE)
        if matches:
            return matches[0][:200]  # Limit title length

    return None

def table_to_dict(table, headers=None):
    """Convert a table to a dictionary format."""
    if not table or len(table) == 0:
        return None

    # Clean all cells
    cleaned_table = []
    for row in table:
        cleaned_row = [clean_text(cell) for cell in row]
        # Skip completely empty rows
        if any(cell for cell in cleaned_row):
            cleaned_table.append(cleaned_row)

    if not cleaned_table:
        return None

    # Try to identify header row (first non-empty row)
    if headers is None and len(cleaned_table) > 0:
        headers = cleaned_table[0]
        data_rows = cleaned_table[1:]
    else:
        data_rows = cleaned_table

    return {
        "headers": headers,
        "rows": data_rows,
        "raw": cleaned_table
    }

def extract_tables_from_pdf(pdf_path, code_name):
    """Extract all tables from a PDF file."""
    print(f"\nProcessing: {pdf_path}")
    print(f"Code: {code_name}")
    print("-" * 50)

    tables_data = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"Total pages: {total_pages}")

            tables_found = 0

            for page_num, page in enumerate(pdf.pages, 1):
                # Progress indicator - print every 10 pages for better monitoring
                if page_num % 10 == 0 or page_num == 1:
                    print(f"  Processing page {page_num}/{total_pages}... (tables found so far: {tables_found})", flush=True)

                try:
                    # Extract tables from page
                    page_tables = page.extract_tables()
                    page_text = page.extract_text() or ""

                    for table_idx, table in enumerate(page_tables):
                        if table and len(table) > 1:  # At least header + 1 row
                            table_dict = table_to_dict(table)

                            if table_dict and len(table_dict["raw"]) > 1:
                                # Try to find table title
                                title = extract_table_title(page, None, page_text)

                                # Create table entry
                                table_entry = {
                                    "id": f"{code_name}_p{page_num}_t{table_idx}",
                                    "code": code_name,
                                    "page": page_num,
                                    "table_index": table_idx,
                                    "title": title,
                                    "headers": table_dict["headers"],
                                    "rows": table_dict["rows"],
                                    "row_count": len(table_dict["rows"]),
                                    "col_count": len(table_dict["headers"]) if table_dict["headers"] else 0,
                                    "raw_content": table_dict["raw"],
                                    # Create searchable text version
                                    "searchable_text": " | ".join([
                                        " ".join(row) for row in table_dict["raw"]
                                    ])
                                }

                                tables_data.append(table_entry)
                                tables_found += 1

                except Exception as e:
                    print(f"  Warning: Error on page {page_num}: {str(e)[:50]}")
                    continue

            print(f"Tables extracted: {tables_found}")

    except Exception as e:
        print(f"Error processing PDF: {e}")
        return []

    return tables_data

def main():
    print("=" * 60)
    print("TABLE EXTRACTION FROM NSR-10 AND ACI-318")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")

    pdf_files = [
        ("pdfs/NSR-10.pdf", "NSR-10"),
        # ("pdfs/ACI-318-19.pdf", "ACI-318"),  # Skipped - large file
    ]

    all_tables = []

    for pdf_path, code_name in pdf_files:
        if os.path.exists(pdf_path):
            tables = extract_tables_from_pdf(pdf_path, code_name)
            all_tables.extend(tables)
        else:
            print(f"Warning: {pdf_path} not found")

    # Save to JSON
    output_file = "tables_extracted.json"

    output_data = {
        "extraction_date": datetime.now().isoformat(),
        "total_tables": len(all_tables),
        "by_code": {
            "NSR-10": len([t for t in all_tables if t["code"] == "NSR-10"]),
            "ACI-318": len([t for t in all_tables if t["code"] == "ACI-318"]),
        },
        "tables": all_tables
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total tables extracted: {len(all_tables)}")
    print(f"  - NSR-10: {output_data['by_code']['NSR-10']}")
    print(f"  - ACI-318: {output_data['by_code']['ACI-318']}")
    print(f"Output saved to: {output_file}")
    print(f"Finished: {datetime.now().isoformat()}")

    # Show sample
    if all_tables:
        print("\n" + "=" * 60)
        print("SAMPLE TABLES")
        print("=" * 60)
        for table in all_tables[:3]:
            print(f"\n[{table['code']} - Page {table['page']}]")
            if table['title']:
                print(f"Title: {table['title']}")
            print(f"Size: {table['row_count']} rows x {table['col_count']} cols")
            if table['headers']:
                print(f"Headers: {table['headers'][:5]}...")  # First 5 headers

if __name__ == "__main__":
    main()
