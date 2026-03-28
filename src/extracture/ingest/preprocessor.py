"""Image preprocessing pipeline for improving OCR accuracy.

Research-backed: deskewing (5-10%), upscaling to 300 DPI (3-8%),
CLAHE contrast enhancement (2-5%), denoising (2-3%).
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

from extracture.config import ExtractureConfig, get_config

logger = logging.getLogger(__name__)


@dataclass
class QualityAssessment:
    """Assessment of document image quality."""

    estimated_dpi: int = 300
    skew_angle: float = 0.0
    contrast_score: float = 1.0
    noise_level: float = 0.0
    is_photo: bool = False
    needs_preprocessing: bool = False
    details: dict = field(default_factory=dict)


class Preprocessor:
    """Applies research-backed preprocessing steps to improve OCR accuracy."""

    def __init__(self, config: ExtractureConfig | None = None):
        self.config = config or get_config()

    def assess_quality(self, image_bytes: bytes) -> QualityAssessment:
        """Assess image quality to determine which preprocessing steps are needed."""
        try:
            import numpy as np
            from PIL import Image

            img = Image.open(io.BytesIO(image_bytes))
            arr = np.array(img.convert("L"))  # Grayscale

            # Estimate contrast
            contrast = float(arr.std() / 128.0)  # Normalized 0-2, 1.0 = good

            # Estimate noise (using Laplacian variance)
            # High variance in Laplacian = sharp, low = blurry or noisy
            from PIL import ImageFilter

            laplacian = img.convert("L").filter(ImageFilter.FIND_EDGES)
            lap_arr = np.array(laplacian)
            sharpness = float(lap_arr.var())

            # Noise estimation: if many isolated pixels differ significantly
            noise_level = 0.0
            if sharpness < 100:
                noise_level = 0.5
            elif sharpness < 500:
                noise_level = 0.2

            # Estimate DPI from image size (heuristic for letter-size documents)
            width, height = img.size
            # Standard US letter = 8.5 x 11 inches
            estimated_dpi = max(int(min(width / 8.5, height / 11.0)), 72)

            # Skew detection (simplified)
            skew_angle = self._detect_skew(arr)

            needs_preprocessing = (
                estimated_dpi < self.config.min_dpi_threshold
                or abs(skew_angle) > self.config.skew_correction_threshold
                or contrast < self.config.contrast_threshold
                or noise_level > 0.3
            )

            return QualityAssessment(
                estimated_dpi=estimated_dpi,
                skew_angle=skew_angle,
                contrast_score=min(contrast, 2.0),
                noise_level=noise_level,
                is_photo=False,
                needs_preprocessing=needs_preprocessing,
                details={
                    "width": width,
                    "height": height,
                    "sharpness": sharpness,
                },
            )
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return QualityAssessment()

    def preprocess(
        self, image_bytes: bytes, quality: QualityAssessment
    ) -> tuple[bytes, list[str]]:
        """Apply preprocessing steps based on quality assessment. Returns (processed_bytes, steps_applied)."""
        if not quality.needs_preprocessing:
            return image_bytes, []

        try:
            from PIL import Image

            img = Image.open(io.BytesIO(image_bytes))
            steps: list[str] = []

            # 1. Deskew (highest impact: 5-10% accuracy gain)
            if abs(quality.skew_angle) > self.config.skew_correction_threshold:
                img = img.rotate(
                    -quality.skew_angle,
                    resample=Image.BICUBIC,
                    expand=True,
                    fillcolor="white" if img.mode == "RGB" else 255,
                )
                steps.append(f"deskew({quality.skew_angle:.1f}deg)")
                logger.debug(f"Deskewed by {quality.skew_angle:.1f} degrees")

            # 2. Upscale to 300 DPI (3-8% accuracy gain)
            if quality.estimated_dpi < self.config.min_dpi_threshold:
                scale = 300 / max(quality.estimated_dpi, 72)
                new_size = (int(img.width * scale), int(img.height * scale))
                img = img.resize(new_size, Image.LANCZOS)
                steps.append(f"upscale({quality.estimated_dpi}→300dpi)")
                logger.debug(f"Upscaled from ~{quality.estimated_dpi} to ~300 DPI")

            # 3. Contrast enhancement with CLAHE (2-5% accuracy gain)
            if quality.contrast_score < self.config.contrast_threshold:
                img = self._apply_clahe(img)
                steps.append("clahe_contrast")
                logger.debug("Applied CLAHE contrast enhancement")

            # 4. Denoise (2-3% accuracy gain on noisy docs)
            if quality.noise_level > 0.3:
                from PIL import ImageFilter

                img = img.filter(ImageFilter.MedianFilter(size=3))
                steps.append("median_denoise")
                logger.debug("Applied median denoising")

            # Convert back to bytes
            buf = io.BytesIO()
            if img.mode == "RGBA":
                img = img.convert("RGB")
            img.save(buf, format="JPEG", quality=95)
            return buf.getvalue(), steps

        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}, returning original")
            return image_bytes, []

    def _detect_skew(self, gray_array) -> float:
        """Detect document skew angle using projection profile method."""
        try:
            import numpy as np

            # Simple projection profile approach
            # Threshold to binary
            threshold = np.mean(gray_array)
            binary = (gray_array < threshold).astype(np.uint8)

            # Try small rotations and find the one with sharpest horizontal projection
            best_angle = 0.0
            best_variance = 0.0

            for angle_10x in range(-100, 101, 5):  # -10 to +10 degrees, step 0.5
                angle = angle_10x / 10.0
                if abs(angle) < 0.5:
                    continue

                from PIL import Image

                rotated = Image.fromarray(binary * 255).rotate(angle, expand=False)
                rotated_arr = np.array(rotated) > 128

                # Horizontal projection
                projection = np.sum(rotated_arr, axis=1).astype(float)
                variance = float(np.var(projection))

                if variance > best_variance:
                    best_variance = variance
                    best_angle = angle

            return best_angle
        except Exception:
            return 0.0

    def _apply_clahe(self, img):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        try:
            import numpy as np
            from PIL import ImageOps

            # Use PIL's autocontrast as a simpler alternative when cv2 is unavailable
            try:
                import cv2

                gray = np.array(img.convert("L"))
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)

                if img.mode == "RGB":
                    # Apply to L channel in LAB space
                    lab = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)
                    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                    from PIL import Image

                    return Image.fromarray(result)
                else:
                    from PIL import Image

                    return Image.fromarray(enhanced)
            except ImportError:
                # Fallback to PIL autocontrast
                return ImageOps.autocontrast(img, cutoff=1)

        except Exception as e:
            logger.warning(f"CLAHE failed: {e}")
            return img
