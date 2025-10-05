from __future__ import annotations

from collections import Counter
from typing import Dict, Sequence

import numpy as np

from engine.diagnostics import radial_spectrum


def compute_symbol_metrics(symbols: Sequence[int]) -> Dict[str, float]:
    counts = Counter(symbols)
    total = sum(counts.values())
    if total == 0:
        return {"entropy": 0.0, "perplexity": 0.0, "rate": 0.0}
    probs = np.array([count / total for count in counts.values()])
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    perplexity = float(2 ** entropy)
    rate = total
    return {"entropy": float(entropy), "perplexity": perplexity, "rate": float(rate)}


def spectrum_metrics(state: np.ndarray) -> Dict[str, np.ndarray]:
    freqs, spec = radial_spectrum(state)
    return {"freqs": freqs, "spectrum": spec}
