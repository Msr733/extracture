"""Confidence calibration using temperature scaling.

Research: uncalibrated models have ECE 0.10-0.20. After temperature scaling: 0.02-0.05.
Calibration must be done per-field, not globally (Guo et al., 2017).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from extracture.config import ExtractureConfig, get_config

logger = logging.getLogger(__name__)


class ConfidenceCalibrator:
    """Per-field temperature scaling for confidence calibration."""

    def __init__(self, config: ExtractureConfig | None = None):
        self.config = config or get_config()
        # Per-field temperature parameters (T > 1 reduces overconfidence)
        self.temperatures: dict[str, float] = {}
        # Global default temperature
        self.default_temperature: float = self.config.default_temperature

    def calibrate(self, field_name: str, raw_confidence: float) -> float:
        """Apply temperature scaling to a raw confidence score."""
        if raw_confidence <= 0.0:
            return 0.0
        if raw_confidence >= 1.0:
            return 1.0

        T = self.temperatures.get(field_name, self.default_temperature)

        if T == 1.0:
            return raw_confidence

        # Convert to logit, scale, convert back
        logit = math.log(raw_confidence / (1.0 - raw_confidence + 1e-10))
        scaled_logit = logit / T
        calibrated = 1.0 / (1.0 + math.exp(-scaled_logit))

        return round(max(0.0, min(1.0, calibrated)), 4)

    def calibrate_fields(
        self, fields: dict[str, Any]
    ) -> dict[str, float]:
        """Calibrate confidence for all fields. Returns field_name -> calibrated_confidence."""
        calibrated = {}
        for field_name, field_result in fields.items():
            raw_conf = (
                field_result.confidence
                if hasattr(field_result, "confidence")
                else float(field_result)
            )
            calibrated[field_name] = self.calibrate(field_name, raw_conf)
        return calibrated

    def fit(
        self,
        validation_data: list[tuple[str, float, bool]],
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> None:
        """Fit temperature parameters using validation data.

        Args:
            validation_data: List of (field_name, predicted_confidence, was_correct)
            lr: Learning rate for gradient descent
            max_iter: Maximum iterations
        """
        # Group by field name
        field_data: dict[str, list[tuple[float, bool]]] = {}
        for field_name, conf, correct in validation_data:
            field_data.setdefault(field_name, []).append((conf, correct))

        for field_name, data in field_data.items():
            T = self._fit_temperature(data, lr=lr, max_iter=max_iter)
            self.temperatures[field_name] = T
            logger.info(f"Calibrated {field_name}: T={T:.3f}")

    def _fit_temperature(
        self,
        data: list[tuple[float, bool]],
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """Fit a single temperature parameter using NLL minimization."""
        T = 1.5  # Start slightly overconfidence-correcting

        for _ in range(max_iter):
            grad = 0.0
            for conf, correct in data:
                if conf <= 0.0 or conf >= 1.0:
                    continue

                logit = math.log(conf / (1.0 - conf + 1e-10))
                scaled_logit = logit / T
                calibrated = 1.0 / (1.0 + math.exp(-scaled_logit))

                # Gradient of NLL w.r.t. T
                target = 1.0 if correct else 0.0
                grad += (calibrated - target) * (-logit / (T * T)) * calibrated * (1 - calibrated)

            grad /= len(data)
            T -= lr * grad
            T = max(0.1, min(10.0, T))  # Clamp

        return round(T, 4)

    def compute_ece(
        self, predictions: list[tuple[float, bool]], n_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error.

        Good calibration: ECE < 0.05
        Poor calibration: ECE > 0.10
        """
        if not predictions:
            return 0.0

        bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
        for conf, correct in predictions:
            bin_idx = min(int(conf * n_bins), n_bins - 1)
            bins[bin_idx].append((conf, correct))

        ece = 0.0
        total = len(predictions)

        for bin_data in bins:
            if not bin_data:
                continue
            avg_conf = sum(c for c, _ in bin_data) / len(bin_data)
            avg_acc = sum(1.0 for _, correct in bin_data if correct) / len(bin_data)
            ece += (len(bin_data) / total) * abs(avg_conf - avg_acc)

        return round(ece, 4)

    def save(self, path: str | Path) -> None:
        """Save calibration parameters to file."""
        data = {
            "default_temperature": self.default_temperature,
            "temperatures": self.temperatures,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str | Path) -> None:
        """Load calibration parameters from file."""
        data = json.loads(Path(path).read_text())
        self.default_temperature = data.get("default_temperature", 1.5)
        self.temperatures = data.get("temperatures", {})
