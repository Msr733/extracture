"""PDF parsing with crash isolation via subprocess."""

from __future__ import annotations

import base64
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from extracture.config import ExtractureConfig, get_config
from extracture.models import PageDimensions, WordPosition

logger = logging.getLogger(__name__)


class PDFProcessingError(Exception):
    pass


class PDFParser:
    """Crash-isolated PDF parser using PyMuPDF in subprocess."""

    def __init__(self, config: ExtractureConfig | None = None):
        self.config = config or get_config()
        self._worker_path = Path(__file__).parent / "_pdf_worker.py"

    def extract_text(
        self, file_bytes: bytes
    ) -> tuple[str | None, list[WordPosition], list[PageDimensions], int]:
        """Extract text, word positions, and page dimensions from a digital PDF."""
        try:
            result = self._run_worker("extract_text", file_bytes)
        except PDFProcessingError:
            logger.warning("PDF text extraction failed, returning empty")
            return None, [], [], 0

        text = result.get("text")
        page_count = result.get("page_count", 0)

        word_positions = []
        for wp in result.get("word_positions", []):
            word_positions.append(
                WordPosition(
                    text=wp["text"],
                    page=wp["page"],
                    x0=wp["x0"],
                    y0=wp["y0"],
                    x1=wp["x1"],
                    y1=wp["y1"],
                    font_name=wp.get("font_name"),
                    font_size=wp.get("font_size"),
                )
            )

        page_dims = []
        for pd in result.get("page_dims", []):
            page_dims.append(
                PageDimensions(page=pd["page"], width=pd["width"], height=pd["height"])
            )

        return text, word_positions, page_dims, page_count

    def render_pages(
        self, file_bytes: bytes, max_pages: int = 50, dpi: int = 300
    ) -> list[bytes]:
        """Render PDF pages to JPEG images."""
        try:
            result = self._run_worker(
                "render_pages",
                file_bytes,
                params={"max_pages": max_pages, "dpi": dpi},
            )
        except PDFProcessingError:
            logger.warning("PDF page rendering failed")
            return []

        images = []
        for b64_img in result.get("images", []):
            images.append(base64.b64decode(b64_img))
        return images

    def get_page_count(self, file_bytes: bytes) -> int:
        try:
            result = self._run_worker("page_count", file_bytes)
            return result.get("count", 0)
        except PDFProcessingError:
            return 0

    def _run_worker(
        self, operation: str, file_bytes: bytes, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Run a PDF operation in an isolated subprocess to prevent SIGSEGV crashes."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as inp:
            inp.write(file_bytes)
            input_path = inp.name

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out:
            output_path = out.name

        try:
            cmd = [
                sys.executable,
                str(self._worker_path),
                operation,
                input_path,
                output_path,
            ]
            if params:
                cmd.append(json.dumps(params))

            env = dict(__import__("os").environ)
            env.pop("PYTHONPATH", None)

            proc = subprocess.run(
                cmd,
                capture_output=True,
                timeout=self.config.subprocess_timeout_seconds,
                env=env,
            )

            if proc.returncode < 0:
                sig = -proc.returncode
                raise PDFProcessingError(
                    f"PDF worker killed by signal {sig} (SIGSEGV={11})"
                )

            output_file = Path(output_path)
            if not output_file.exists():
                stderr = proc.stderr.decode("utf-8", errors="replace")[:500]
                raise PDFProcessingError(f"PDF worker produced no output: {stderr}")

            envelope = json.loads(output_file.read_text())

            if envelope.get("status") == "error":
                raise PDFProcessingError(f"PDF worker error: {envelope.get('message')}")

            return envelope.get("data", {})

        except subprocess.TimeoutExpired:
            raise PDFProcessingError(
                f"PDF worker timed out after {self.config.subprocess_timeout_seconds}s"
            )
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)
