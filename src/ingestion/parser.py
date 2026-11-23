from pathlib import Path
import json
import fitz

RAW_DIR=Path('data/raw')
PROCESSED_DIR=Path('data/processed')
OUTPUT_FILE=PROCESSED_DIR/'pages.jsonl'
COMPANY_NAMES={
    'microsoft':'Microsoft',
    'apple':'Apple',
    'nvidia':'NVIDIA'
}

def parse_pdf(pdf_path: Path)-> list[dict]:
    """
    Extract text from a PDF page-by-page and attach metadata.
    """
    company_key=pdf_path.parent.name.lower()
    company=COMPANY_NAMES.get(company_key,company_key.title())
    year=pdf_path.stem
    document=f"{company} {year} Annual Report"

    pages=[]

    doc=fitz.open(pdf_path)

    for page_number,page in enumerate(doc,start=1):
        text=page.get_text('text').strip()
        if not text:
            continue
        page_data={
            'text':text,
            'company':company,
            'year':year,
            'document':document,
            'page':page_number,
            'section':None
        }
        pages.append(page_data)
    doc.close()
    return pages
def main():
    PROCESSED_DIR.mkdir(parents=True,exist_ok=True)

    pdf_files=sorted(RAW_DIR.glob('*/*.pdf'))
    if not pdf_files:
        print("No PDF files found in data/raw/")
        return 
    all_pages=[]

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path}")
        try:
            pages=parse_pdf(pdf_path)
            all_pages.extend(pages)
            print(f"  Extracted {len(pages)} pages")
        except Exception as e:
            print(f"  Error: {e}")
    with open(OUTPUT_FILE,"w",encoding='utf-8') as f:
        for page in all_pages:
            f.write(json.dumps(page,ensure_ascii=False)+'\n')

    print('\nDone!')
    print(f"Total pages extracted: {len(all_pages)}")
    print(f"Output: {OUTPUT_FILE}")
if __name__=='__main__':
    main()