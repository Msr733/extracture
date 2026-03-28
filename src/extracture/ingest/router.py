"""Smart routing for document ingestion — auto-detect format and optimal strategy."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

from extracture.config import ExtractureConfig, get_config
from extracture.ingest.pdf import PDFParser
from extracture.ingest.preprocessor import Preprocessor, QualityAssessment
from extracture.models import ExtractionMethod, IngestResult, PageDimensions, WordPosition

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}
SUPPORTED_DOC_EXTENSIONS = {".docx", ".xlsx", ".csv", ".html", ".htm", ".txt"}


class IngestRouter:
    """Routes documents through the optimal ingestion pipeline."""

    def __init__(self, config: ExtractureConfig | None = None, ocr_engine: str | None = None):
        self.config = config or get_config()
        self.ocr_engine = ocr_engine or self.config.default_ocr_engine
        self.pdf_parser = PDFParser(config=self.config)
        self.preprocessor = Preprocessor(config=self.config)

    def ingest(self, source: str | Path | bytes, file_type: str | None = None) -> IngestResult:
        """Ingest a document and return parsed content ready for extraction."""
        file_bytes, detected_type = self._load_source(source, file_type)

        if detected_type == "pdf":
            return self._ingest_pdf(file_bytes)
        elif detected_type in ("png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp"):
            return self._ingest_image(file_bytes, detected_type)
        elif detected_type in ("txt", "csv"):
            return self._ingest_text(file_bytes)
        else:
            logger.warning(f"Unsupported file type '{detected_type}', attempting as text")
            return self._ingest_text(file_bytes)

    def _load_source(self, source: str | Path | bytes, file_type: str | None) -> tuple[bytes, str]:
        if isinstance(source, bytes):
            return source, file_type or "pdf"

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_bytes = path.read_bytes()
        detected = file_type or path.suffix.lstrip(".").lower()
        return file_bytes, detected

    def _ingest_pdf(self, file_bytes: bytes) -> IngestResult:
        """Ingest a PDF — detect digital vs scanned, extract accordingly."""
        # Try digital text extraction first (free, 100% accurate)
        text_content, word_positions, page_dims, page_count = self.pdf_parser.extract_text(file_bytes)

        preprocessing_applied: list[str] = []

        if text_content and len(text_content.strip()) >= self.config.digital_text_threshold:
            # Digital PDF — direct text extraction
            logger.info(f"Digital PDF detected ({len(text_content)} chars, {page_count} pages)")
            page_images = self.pdf_parser.render_pages(file_bytes, max_pages=page_count)

            return IngestResult(
                file_type="pdf",
                extraction_method=ExtractionMethod.DIGITAL,
                text_content=text_content,
                word_positions=word_positions,
                page_images=page_images,
                page_dims=page_dims,
                page_count=page_count,
                preprocessing_applied=preprocessing_applied,
            )

        # Scanned PDF — render to images and OCR
        logger.info(f"Scanned PDF detected ({page_count} pages), routing to OCR")
        page_images = self.pdf_parser.render_pages(
            file_bytes, max_pages=page_count, dpi=self.config.ocr_dpi
        )

        # Preprocess if enabled
        if self.config.enable_preprocessing and page_images:
            processed_images = []
            for img_bytes in page_images:
                quality = self.preprocessor.assess_quality(img_bytes)
                processed, steps = self.preprocessor.preprocess(img_bytes, quality)
                processed_images.append(processed)
                preprocessing_applied.extend(steps)
            page_images = processed_images

        # OCR
        ocr_text, ocr_words = self._run_ocr(page_images)

        return IngestResult(
            file_type="pdf",
            extraction_method=ExtractionMethod.SCANNED,
            text_content=ocr_text if ocr_text else None,
            word_positions=ocr_words,
            page_images=page_images,
            page_dims=page_dims,
            page_count=page_count,
            preprocessing_applied=list(set(preprocessing_applied)),
            ocr_engine_used=self.ocr_engine,
        )

    def _ingest_image(self, file_bytes: bytes, file_type: str) -> IngestResult:
        """Ingest an image — preprocess and OCR."""
        preprocessing_applied: list[str] = []

        if self.config.enable_preprocessing:
            quality = self.preprocessor.assess_quality(file_bytes)
            file_bytes, preprocessing_applied = self.preprocessor.preprocess(file_bytes, quality)

        ocr_text, ocr_words = self._run_ocr([file_bytes])

        return IngestResult(
            file_type=file_type,
            extraction_method=ExtractionMethod.SCANNED,
            text_content=ocr_text if ocr_text else None,
            word_positions=ocr_words,
            page_images=[file_bytes],
            page_dims=[],
            page_count=1,
            preprocessing_applied=preprocessing_applied,
            ocr_engine_used=self.ocr_engine,
        )

    def _ingest_text(self, file_bytes: bytes) -> IngestResult:
        """Ingest a plain text file."""
        text = file_bytes.decode("utf-8", errors="replace")
        return IngestResult(
            file_type="txt",
            extraction_method=ExtractionMethod.DIGITAL,
            text_content=text,
            page_count=1,
        )

    def _run_ocr(self, page_images: list[bytes]) -> tuple[str | None, list[WordPosition]]:
        """Run OCR using the configured engine."""
        if self.ocr_engine == "pymupdf":
            # PyMuPDF doesn't do OCR — return empty for scanned docs
            return None, []

        try:
            if self.ocr_engine == "tesseract":
                return self._ocr_tesseract(page_images)
            elif self.ocr_engine == "surya":
                return self._ocr_surya(page_images)
            elif self.ocr_engine == "paddleocr":
                return self._ocr_paddleocr(page_images)
            elif self.ocr_engine == "doctr":
                return self._ocr_doctr(page_images)
            else:
                logger.warning(f"Unknown OCR engine '{self.ocr_engine}', skipping OCR")
                return None, []
        except ImportError as e:
            logger.error(f"OCR engine '{self.ocr_engine}' not installed: {e}")
            logger.info(f"Install with: pip install extracture[{self.ocr_engine}]")
            return None, []

    def _ocr_tesseract(self, page_images: list[bytes]) -> tuple[str, list[WordPosition]]:
        import pytesseract
        from PIL import Image

        all_text: list[str] = []
        all_words: list[WordPosition] = []

        for page_num, img_bytes in enumerate(page_images):
            img = Image.open(io.BytesIO(img_bytes))
            width, height = img.size

            text = pytesseract.image_to_string(img)
            all_text.append(text)

            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            for i, word in enumerate(data["text"]):
                if not word.strip():
                    continue
                conf = float(data["conf"][i])
                if conf < 0:
                    continue
                all_words.append(
                    WordPosition(
                        text=word,
                        page=page_num,
                        x0=data["left"][i] / width,
                        y0=data["top"][i] / height,
                        x1=(data["left"][i] + data["width"][i]) / width,
                        y1=(data["top"][i] + data["height"][i]) / height,
                        confidence=conf / 100.0,
                    )
                )

        return "\n\n".join(all_text), all_words

    def _ocr_surya(self, page_images: list[bytes]) -> tuple[str, list[WordPosition]]:
        from PIL import Image

        try:
            from surya.recognition import RecognitionPredictor
            from surya.detection import DetectionPredictor

            det_predictor = DetectionPredictor()
            rec_predictor = RecognitionPredictor()
        except ImportError:
            from surya.ocr import run_ocr
            from surya.model.detection.model import load_model as load_det_model
            from surya.model.recognition.model import load_model as load_rec_model

            det_model = load_det_model()
            rec_model = load_rec_model()

            images = [Image.open(io.BytesIO(b)) for b in page_images]
            results = run_ocr(images, det_model=det_model, rec_model=rec_model)

            all_text: list[str] = []
            all_words: list[WordPosition] = []

            for page_num, result in enumerate(results):
                page_text_parts: list[str] = []
                img = images[page_num]
                w, h = img.size

                for line in result.text_lines:
                    page_text_parts.append(line.text)
                    bbox = line.bbox
                    all_words.append(
                        WordPosition(
                            text=line.text,
                            page=page_num,
                            x0=bbox[0] / w,
                            y0=bbox[1] / h,
                            x1=bbox[2] / w,
                            y1=bbox[3] / h,
                            confidence=line.confidence,
                        )
                    )
                all_text.append("\n".join(page_text_parts))

            return "\n\n".join(all_text), all_words

        # Newer Surya API
        images = [Image.open(io.BytesIO(b)) for b in page_images]
        all_text_parts: list[str] = []
        all_words_list: list[WordPosition] = []

        for page_num, img in enumerate(images):
            w, h = img.size
            det_result = det_predictor([img])
            rec_result = rec_predictor([img], det_result)

            page_parts: list[str] = []
            for line in rec_result[0].text_lines:
                page_parts.append(line.text)
                bbox = line.bbox
                all_words_list.append(
                    WordPosition(
                        text=line.text,
                        page=page_num,
                        x0=bbox[0] / w,
                        y0=bbox[1] / h,
                        x1=bbox[2] / w,
                        y1=bbox[3] / h,
                        confidence=getattr(line, "confidence", 0.9),
                    )
                )
            all_text_parts.append("\n".join(page_parts))

        return "\n\n".join(all_text_parts), all_words_list

    def _ocr_paddleocr(self, page_images: list[bytes]) -> tuple[str, list[WordPosition]]:
        from paddleocr import PaddleOCR
        from PIL import Image
        import numpy as np

        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

        all_text: list[str] = []
        all_words: list[WordPosition] = []

        for page_num, img_bytes in enumerate(page_images):
            img = Image.open(io.BytesIO(img_bytes))
            w, h = img.size
            img_array = np.array(img)

            result = ocr.ocr(img_array, cls=True)
            page_parts: list[str] = []

            if result and result[0]:
                for line in result[0]:
                    box, (text, conf) = line[0], line[1]
                    page_parts.append(text)
                    x_coords = [p[0] for p in box]
                    y_coords = [p[1] for p in box]
                    all_words.append(
                        WordPosition(
                            text=text,
                            page=page_num,
                            x0=min(x_coords) / w,
                            y0=min(y_coords) / h,
                            x1=max(x_coords) / w,
                            y1=max(y_coords) / h,
                            confidence=conf,
                        )
                    )

            all_text.append("\n".join(page_parts))

        return "\n\n".join(all_text), all_words

    def _ocr_doctr(self, page_images: list[bytes]) -> tuple[str, list[WordPosition]]:
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor

        predictor = ocr_predictor(pretrained=True)

        all_text: list[str] = []
        all_words: list[WordPosition] = []

        for page_num, img_bytes in enumerate(page_images):
            doc = DocumentFile.from_images([img_bytes])
            result = predictor(doc)

            page_parts: list[str] = []
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        line_text = " ".join(w.value for w in line.words)
                        page_parts.append(line_text)
                        for word in line.words:
                            geo = word.geometry
                            all_words.append(
                                WordPosition(
                                    text=word.value,
                                    page=page_num,
                                    x0=geo[0][0],
                                    y0=geo[0][1],
                                    x1=geo[1][0],
                                    y1=geo[1][1],
                                    confidence=word.confidence,
                                )
                            )

            all_text.append("\n".join(page_parts))

        return "\n\n".join(all_text), all_words
