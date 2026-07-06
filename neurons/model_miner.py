"""Poker44 miner backed by the trained LightGBM bot-detector.

Drop-in replacement for neurons/miner.py. It loads the model produced by
poker44_model/train.py and returns one calibrated risk score per chunk.

Run it the same way as the reference miner, e.g.:
    POKER44_MODEL_DIR=/home/allice/poker44_model \
    WALLET_NAME=my_cold HOTKEY=my_hot AXON_PORT=8091 \
    python neurons/model_miner.py --netuid 126 --subtensor.network finney ...
"""

# from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse

# Make the trained-model package importable (features.py + serve.py + model/).
_MODEL_PKG = os.getenv("POKER44_MODEL_DIR", "/home/allice/poker44_model")
sys.path.insert(0, _MODEL_PKG)
_VER = os.getenv("POKER44_MODEL_VERSION", "comp").strip().lower()
if _VER == "si":
    from serve_si import ChunkScorerSI as ChunkScorer  # noqa: E402
elif _VER == "comp":
    from serve_comp import ChunkScorerComp as ChunkScorer  # noqa: E402
elif _VER == "v3":
    from serve_v3 import ChunkScorerV3 as ChunkScorer  # noqa: E402
else:
    from serve import ChunkScorer  # noqa: E402


class ModelMiner(BaseMinerNeuron):
    def __init__(self, config=None):
        super(ModelMiner, self).__init__(config=config)
        self.scorer = ChunkScorer()
        bt.logging.info(
            f"🤖 Poker44 LightGBM miner | threshold={self.scorer.threshold:.3f} "
            f"cv_ap={self.scorer.meta.get('cv_ap')} cv_auc={self.scorer.meta.get('cv_auc')}"
        )

        repo_root = Path(__file__).resolve().parents[1]
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=[
                Path(__file__).resolve(),
                Path(_MODEL_PKG) / "features.py",
                Path(_MODEL_PKG) / "serve.py",
            ],
            defaults={
                "model_name": "poker44-lgbm-dispersion",
                "model_version": "1",
                "framework": "lightgbm",
                "license": "MIT",
                # TODO: point these at YOUR public repo + real commit before
                # chasing top scores, or the manifest audit can zero you out.
                "repo_url": "https://github.com/Poker44/Poker44-subnet",
                "notes": "GBDT over within-group dispersion features.",
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Trained on the public Poker44 benchmark releases only."
                ),
                "training_data_sources": ["poker44-public-benchmark"],
                "private_data_attestation": (
                    "Does not train on validator-only live evaluation data."
                ),
            },
        )
        self.manifest_digest = manifest_digest(self.model_manifest)
        comp = evaluate_manifest_compliance(self.model_manifest)
        bt.logging.info(f"Manifest transparency: {comp['status']} missing={comp['missing_fields']}")

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []
        # Accumulate a deduplicated corpus of REAL live chunks (unlabeled) for
        # domain-adaptation / pseudo-label retraining. Capped to bound disk.
        if chunks and os.getenv("POKER44_LOG_LIVE_CORPUS", "1") not in {"0", "false"}:
            try:
                import json, hashlib
                if not hasattr(self, "_corpus_seen"):
                    self._corpus_seen = set()
                    self._corpus_dir = "/home/allice/poker44_model/live_corpus"
                    os.makedirs(self._corpus_dir, exist_ok=True)
                    self._corpus_cap = int(os.getenv("POKER44_LIVE_CORPUS_CAP", "1500"))
                h = hashlib.sha256(
                    json.dumps(chunks, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()[:16]
                if h not in self._corpus_seen and len(self._corpus_seen) < self._corpus_cap:
                    self._corpus_seen.add(h)
                    with open(os.path.join(self._corpus_dir, f"{h}.json"), "w") as fh:
                        json.dump(chunks, fh)
                    if len(self._corpus_seen) % 25 == 0:
                        bt.logging.info(f"live corpus: {len(self._corpus_seen)} unique batches saved")
            except Exception as exc:
                bt.logging.warning(f"corpus log failed: {exc}")
        try:
            scores = self.scorer.score_chunks(chunks)
        except Exception as exc:  # never crash the axon on a bad payload
            bt.logging.warning(f"scoring failed ({exc}); returning neutral 0.5s")
            scores = [0.5] * len(chunks)
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        if scores:
            srt = sorted(scores)
            mn, mx = srt[0], srt[-1]
            mean = sum(scores) / len(scores)
            med = srt[len(srt) // 2]
            distinct = len(set(round(s, 4) for s in scores))
            bt.logging.info(
                f"Scored {len(chunks)} chunks | bots>=0.5: {sum(s>=0.5 for s in scores)} | "
                f"score min={mn:.3f} mean={mean:.3f} med={med:.3f} max={mx:.3f} distinct={distinct}"
            )
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with ModelMiner() as miner:
        bt.logging.info("Poker44 LightGBM miner running...")
        while True:
            bt.logging.info(f"UID {miner.uid} | Incentive {miner.metagraph.I[miner.uid]}")
            time.sleep(5 * 60)
