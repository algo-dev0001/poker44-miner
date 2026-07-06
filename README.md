# Poker44 SN126 — Bot-Detection Miner

A supervised bot-detection model for the Poker44 Bittensor subnet (netuid 126).
It scores `DetectionSynapse` chunks with a per-chunk bot-risk score in `[0,1]`,
optimized for the subnet's ranking-based reward (average precision + recall at a
bounded false-positive rate).

## Model

- **Type:** stacked ensemble — LightGBM + XGBoost + CatBoost + ExtraTrees +
  RandomForest, with isotonic calibration.
- **Features:** scale-invariant chunk features (action-mix ratios, entropies,
  action/actor/street signatures, quantiles) — absolute big-blind magnitudes are
  dropped so the model generalizes across stack depths.
- **Training data:** the **public Poker44 training benchmark** only, fetched from
  `https://api.poker44.net/api/v1/benchmark`. No validator-only / live evaluation
  data is used in training.
- **Serving:** raw calibrated scores are returned directly (ranking-first); no
  fixed per-batch proportion is imposed.

## Attribution

This implementation is built on the open-source (MIT-licensed) Poker44 model
pipeline published by the community:

- `poker44_ml/` (features, stacked model, calibration, inference) is derived from
  **[Travis861/Poker44_v1](https://github.com/Travis861/Poker44_v1)** (MIT).
- Calibration/inference structure also informed by
  **[phasberg/poker44-model](https://github.com/phasberg)** (MIT).

Our contribution is the training configuration (scale-invariant feature subset,
benchmark v2.2 dataset assembly) and the serving wrapper (`serve_si.py`).

## Files

- `poker44_ml/` — feature extraction, stacked model, calibration, inference
- `serve_si.py` — serving wrapper (`ChunkScorerSI`) used by the miner
- `neurons/model_miner.py` — the Bittensor miner
- `models/` — trained model artifact (`.joblib`)

## Reproduce

```bash
pip install -r requirements.txt
# fetch the public benchmark, then:
python -m training.train_model_v2 --benchmark-path <benchmark.txt> \
  --output models/model.joblib --robust-features-only
```

## License

MIT — see `LICENSE`.
