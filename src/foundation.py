"""Zero-shot foundation model wrappers for the common rolling loop (CLAUDE.md section 6)."""

import numpy as np
import pandas as pd
import torch

from src.config import EXOG, FM_CONTEXT_LENGTH, HORIZON, QUANTILE_COLUMNS, QUANTILES, TARGET

TIMESFM3_CHECKPOINT = "google/timesfm-3.0-pytorch"
TIMESTRAL_CHECKPOINT = "mistralai/Timestral-209M"


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


class Timestral:
    """Timestral-209M (zero-shot), univariate or with the EXOG known-future covariates.

    `model` is a `timestral.TimestralForecaster`; the same loaded model can be shared by the
    univariate and covariate wrappers. `forecast()` is called with its default parameters except
    `clip_nonnegative=False`: the default clamps the forecasts at 0 when the whole context is
    non-negative, and negative prices are real values. Inference is deterministic. The point
    forecast is the median (quantile 0.5); the deciles 0.1 ... 0.9 are exact levels of the 21.
    """

    def __init__(self, model, with_covariates: bool, context_length: int = FM_CONTEXT_LENGTH):
        levels = [round(float(q), 2) for q in model.quantile_levels]
        missing = [q for q in QUANTILES if q not in levels]
        if missing:
            raise ValueError(f"Timestral quantile levels {levels} do not contain {missing}")
        if context_length > model.max_context:
            raise ValueError(f"context_length {context_length} exceeds the model context {model.max_context}")
        self.model = model
        self.with_covariates = with_covariates
        self.context_length = context_length
        self.quantile_index = [levels.index(q) for q in QUANTILES]

    @staticmethod
    def load(device: str | None = None):
        """Load the pretrained `TimestralForecaster` (209M) on `device` (default: get_device(); the library defaults to CPU)."""
        from timestral import TimestralForecaster

        return TimestralForecaster.from_pretrained(TIMESTRAL_CHECKPOINT, device=device or get_device())

    def predict(self, history: pd.DataFrame, future_cov: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        context, covariates = fm_inputs(history, future_cov, self.context_length)
        fc = self.model.forecast(
            context[None, None],  # (B=1, V=1, T)
            prediction_length=HORIZON,
            known_covariates=covariates[None] if self.with_covariates else None,  # (1, n_cov, T + 24)
            clip_nonnegative=False,
        )
        all_quantiles = fc.quantiles[0, 0].float().cpu().numpy()  # (24, 21)
        quantiles = pd.DataFrame(all_quantiles[:, self.quantile_index].astype(float), columns=QUANTILE_COLUMNS)
        median = fc.median[0, 0].float().cpu().numpy().astype(float)
        return median, quantiles
