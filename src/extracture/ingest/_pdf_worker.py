"""Subprocess worker for crash-isolated PDF operations.

This script runs in a separate process so that segfaults in PDF libraries
(PyMuPDF/pdfplumber) cannot kill the main process.

Usage: python _pdf_worker.py <operation> <input_path> <output_path> [params_json]
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

# Remove own directory from path to avoid shadowing stdlib
own_dir = str(Path(__file__).parent)
sys.path = [p for p in sys.path if p != own_dir]


def extract_text(input_path: str) -> dict:
    import fitz  # PyMuPDF

    doc = fitz.open(input_path)
    page_count = len(doc)

    all_text_parts: list[str] = []
    word_positions: list[dict] = []
    page_dims: list[dict] = []

    for page_num in range(page_count):
        page = doc[page_num]
        rect = page.rect
        width, height = rect.width, rect.height

        page_dims.append({"page": page_num, "width": width, "height": height})

        # Extract text with layout preservation
        text = page.get_text("text")
        all_text_parts.append(text)

        # Extract word positions
        words = page.get_text("words")  # (x0, y0, x1, y1, word, block_no, line_no, word_no)
        for w in words:
            x0, y0, x1, y1, word_text = w[0], w[1], w[2], w[3], w[4]
            if not word_text.strip():
                continue
            word_positions.append({
                "text": word_text,
                "page": page_num,
                "x0": round(x0 / width, 6) if width > 0 else 0,
                "y0": round(y0 / height, 6) if height > 0 else 0,
                "x1": round(x1 / width, 6) if width > 0 else 0,
                "y1": round(y1 / height, 6) if height > 0 else 0,
            })

    doc.close()

    return {
        "text": "\n\n".join(all_text_parts),
        "word_positions": word_positions,
        "page_dims": page_dims,
        "page_count": page_count,
    }


def render_pages(input_path: str, params: dict) -> dict:
    import fitz

    max_pages = params.get("max_pages", 50)
    dpi = params.get("dpi", 300)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    doc = fitz.open(input_path)
    page_count = min(len(doc), max_pages)
    images: list[str] = []

    for page_num in range(page_count):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("jpeg")
        images.append(base64.b64encode(img_bytes).decode("ascii"))

    doc.close()
    return {"images": images, "page_count": page_count}


def page_count(input_path: str) -> dict:
    import fitz

    doc = fitz.open(input_path)
    count = len(doc)
    doc.close()
    return {"count": count}


OPERATIONS = {
    "extract_text": lambda inp, params: extract_text(inp),
    "render_pages": render_pages,
    "page_count": lambda inp, params: page_count(inp),
}


def main():
    if len(sys.argv) < 4:
        print("Usage: _pdf_worker.py <operation> <input_path> <output_path> [params_json]", file=sys.stderr)
        sys.exit(1)

    operation = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3]
    params = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}

    if operation not in OPERATIONS:
        result = {"status": "error", "message": f"Unknown operation: {operation}"}
    else:
        try:
            data = OPERATIONS[operation](input_path, params)
            result = {"status": "ok", "data": data}
        except Exception as e:
            result = {"status": "error", "message": str(e)}

    Path(output_path).write_text(json.dumps(result))


if __name__ == "__main__":
    main()
