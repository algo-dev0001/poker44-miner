"""Serve the SCALE-INVARIANT model (magnitude features dropped) + rank-norm.

The scale-invariant stacked model de-saturates on live (spreads its scores) but
its absolute calibration is off (everything lands >0.5). We keep its *ranking*
(which is the experiment) and bolt a conservative per-batch rank normalization on
top to place the 0.5 cut so only the top fraction are called bot -> FPR-safe if
the ranking has real signal.
"""
from __future__ import annotations
import os, sys, json
from typing import Any, Dict, List
import numpy as np

_COMP = "/home/allice/competitor_travis"
if _COMP not in sys.path:
    sys.path.append(_COMP)
from poker44_ml.inference import Poker44Model  # noqa: E402

_MODEL = os.getenv("POKER44_SI_MODEL", os.path.join(_COMP, "models", "poker44_stacked_scaleinv.joblib"))


class ChunkScorerSI:
    def __init__(self, model_path: str = _MODEL):
        self.predictor = Poker44Model(model_path)
        self.meta = dict(getattr(self.predictor, "metadata", {}) or {})
        self.bot_fraction = float(os.getenv("POKER44_BOT_FRACTION", "0.15"))
        self.rank_min_batch = int(os.getenv("POKER44_RANK_MIN_BATCH", "6"))
        self.threshold = self.bot_fraction

    def score_chunks(self, chunks: List[List[Dict[str, Any]]]) -> List[float]:
        if not chunks:
            return []
        raw = np.asarray(self.predictor.predict_chunk_scores(chunks), dtype=float)
        # New reward (2026-07) is pure RANKING (0.75*AP + 0.25*recall@FPR<=0.05),
        # computed by sorting scores. Rank-norm forces a fixed per-batch fraction
        # and destroys cross-batch order -> hurts AP. Serve raw calibrated scores.
        if os.getenv("POKER44_RAW", "1") not in {"0", "false", "no"}:
            return [round(float(s), 6) for s in raw]
        n = len(raw)
        if n >= self.rank_min_batch:
            order = raw.argsort()
            ranks = np.empty(n, dtype=float)
            ranks[order] = np.arange(n)
            p = (ranks + 0.5) / n
            f = min(max(self.bot_fraction, 1e-3), 0.999)
            cut = 1.0 - f
            out = np.where(p >= cut, 0.5 + 0.5 * (p - cut) / f, 0.5 * p / max(cut, 1e-6))
        else:
            out = np.clip(raw, 0.0, 1.0)
        return [round(float(s), 6) for s in out]


if __name__ == "__main__":
    s = ChunkScorerSI()
    live = json.load(open(os.path.join(os.path.dirname(__file__), "live_chunk_sample.json")))
    sc = np.array(s.score_chunks(live))
    print(f"SI+ranknorm(f={s.bot_fraction}): bots>=0.5={int((sc>=0.5).sum())}/{len(sc)} "
          f"band[{sc.min():.3f},{sc.max():.3f}]")
