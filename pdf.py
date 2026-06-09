from pypdf import PdfReader
from pathlib import Path


PDF_PATH = Path("data/source.pdf")
TXT_PATH = Path("data/source.txt")

reader = PdfReader(PDF_PATH)
print(f"PDF has {len(reader.pages)} pages")

full_text = []

for i ,page in enumerate(reader.pages, start=1):
        
        text = page.extract_text() or ""
        full_text.append(f"--- Page {i} ---\n{text}")
        combined = "\n".join(full_text)
        TXT_PATH.write_text(combined, encoding="utf-8")
        print("----------------------------")
        print(f"Extracted {len(combined)} characters to {TXT_PATH}")
        print("2----222222222----------------------------")
        print("\n--- First 500 chars ---")
        print("-33333333333---------------------------")
        print("lasttt-----------",combined[:500])



    
