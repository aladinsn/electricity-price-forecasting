"""Zero-shot foundation model wrappers for the common rolling loop (CLAUDE.md section 6)."""

import numpy as np
import pandas as pd
import torch

from src.config import EXOG, FM_CONTEXT_LENGTH, HORIZON, QUANTILE_COLUMNS, QUANTILES, TARGET

TIMESFM3_CHECKPOINT = "google/timesfm-3.0-pytorch"


def get_device() -> str:
    """CUDA if available, else CPU (chosen at runtime, never hard-coded)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def fm_inputs(
    history: pd.DataFrame, future_cov: pd.DataFrame, context_length: int = FM_CONTEXT_LENGTH
) -> tuple[np.ndarray, np.ndarray]:
    """Model inputs for day D, built only from the information available at the origin.

    Returns:
    - context: the last `context_length` prices of `history` (up to D-1 23:00), shape (context_length,);
    - covariates: the EXOG forecasts and calendar over context + D, one row per covariate,
      shape (len(EXOG), context_length + 24). Known-future covariates: the same kind of data
      (forecasts, never actual values) in the past and in the future.
    """
    if len(history) < context_length:
        raise ValueError(f"history has {len(history)} rows, the context needs {context_length}")
    past = history.iloc[-context_length:]
    context = past[TARGET].to_numpy(dtype=np.float32)
    covariates = pd.concat([past[EXOG], future_cov[EXOG]]).to_numpy(dtype=np.float32).T
    assert covariates.shape == (len(EXOG), context_length + HORIZON)
    assert np.isfinite(context).all() and np.isfinite(covariates).all()
    return context, covariates


class TimesFM3:
    """TimesFM 3 (zero-shot), univariate or with the EXOG known-future covariates.

    `model` is a `timesfm.TimesFM3Forecaster` (not the `TimesFM3Evaluator`, whose defaults clip
    forecasts at 0); the same loaded model can be shared by the univariate and covariate wrappers.
    Inference is deterministic. The point forecast is the median (quantile 0.5).
    """

    def __init__(self, model, with_covariates: bool, context_length: int = FM_CONTEXT_LENGTH):
        levels = [round(float(q), 2) for q in model.config.quantiles]
        if levels != QUANTILES:
            raise ValueError(f"Unexpected TimesFM 3 quantile levels: {levels}, expected {QUANTILES}")
        self.model = model
        self.with_covariates = with_covariates
        self.context_length = context_length

    @staticmethod
    def load(device: str | None = None):
        """Load the pretrained `TimesFM3Forecaster` (331M parameters) on `device` (default: get_device())."""
        from timesfm import TimesFM3Forecaster

        return TimesFM3Forecaster.from_pretrained(TIMESFM3_CHECKPOINT, device=device or get_device())

    def predict(self, history: pd.DataFrame, future_cov: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        context, covariates = fm_inputs(history, future_cov, self.context_length)
        out = self.model.predict(
            context,
            horizon=HORIZON,
            past_future_covariates=covariates if self.with_covariates else None,
            return_quantiles=True,
        )
        quantiles = pd.DataFrame(np.asarray(out.quantiles, dtype=float), columns=QUANTILE_COLUMNS)
        return np.asarray(out.forecast, dtype=float), quantiles
