"""Fixed-scale causal transforms; no fitted future baseline."""

from math import isfinite, tanh


def signed_tanh(value: float, scale: float) -> float:
    if not isfinite(value) or not isfinite(scale) or scale <= 0:
        raise ValueError("normalization needs finite value and positive finite scale")
    return tanh(value / scale)
