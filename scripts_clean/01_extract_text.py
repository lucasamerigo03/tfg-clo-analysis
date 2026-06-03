import pdfplumber
import os
from tqdm import tqdm

# Paths relative to the project root
RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)


def extract_text_from_pdf(pdf_path):
    """
    Extracts text from all pages of a PDF.
    Returns the concatenated text as a string, or None if extraction fails.
    """
    pages_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
    except Exception as e:
        print(f"  ERROR reading {os.path.basename(pdf_path)}: {e}")
        return None
    return "\n".join(pages_text)


def main():
    pdf_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".pdf")]
    print(f"PDFs found: {len(pdf_files)}\n")

    success = 0
    failed = []

    for filename in tqdm(pdf_files, desc="Extracting text"):
        pdf_path = os.path.join(RAW_DIR, filename)
        txt_filename = filename.replace(".pdf", ".txt")
        txt_path = os.path.join(PROCESSED_DIR, txt_filename)

        # Skip if the .txt already exists
        if os.path.exists(txt_path):
            success += 1
            continue

        text = extract_text_from_pdf(pdf_path)

        if text and len(text.strip()) > 100:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            success += 1
        else:
            failed.append(filename)

    print(f"\nDone: {success}/{len(pdf_files)} extracted successfully")
    if failed:
        print(f"Failed ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
