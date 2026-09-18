from pathlib import Path
import fitz
import pytesseract
from PIL import Image, ImageOps, ImageEnhance, ImageFilter


def extract_text_from_pdf(pdf_path):
    pdf = fitz.open(pdf_path)
    documents = []

    for page_number, page in enumerate(pdf, start=1):
        print(f"Processing page {page_number}/{len(pdf)}...")

        # Render page at high resolution
        pix = page.get_pixmap(
            matrix=fitz.Matrix(3, 3),
            alpha=False
        )

        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples
        )

        # Convert to grayscale
        image = ImageOps.grayscale(image)

        # Improve contrast
        image = ImageOps.autocontrast(image)

        # Sharpen the image
        image = image.filter(ImageFilter.SHARPEN)

        # Increase contrast slightly
        image = ImageEnhance.Contrast(image).enhance(1.5)

        # OCR
        text = pytesseract.image_to_string(
            image,
            config="--psm 3 -c preserve_interword_spaces=1"
        )

        if text.strip():
            documents.append({
                "page": page_number,
                "text": text
            })

    pdf.close()
    return documents


if __name__ == "__main__":
    pdf_path = Path("data/documents/sample.pdf")

    documents = extract_text_from_pdf(pdf_path)

    print("\n================================")
    print("OCR COMPLETED")
    print("================================")
    print(f"Pages with text: {len(documents)}")

    # Save OCR text
    output_path = Path("data/extracted_text.txt")

    with open(output_path, "w", encoding="utf-8") as file:
        for document in documents:
            file.write(f"\n--- Page {document['page']} ---\n")
            file.write(document["text"])
            file.write("\n")

    print(f"\nText saved to: {output_path}")

    # Show first page
    if documents:
        print("\n--- Page 1 ---")
        print(documents[0]["text"][:1500])