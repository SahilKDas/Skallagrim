# Initial trained checkpoints

These are reproducible bootstrap candidates, not established competitive bots.

Both checkpoints use 1,000 completed training games and 250 held-out games from audited baseline self-play (8 ms per move, exploration 0.2, seed 20261009). Training retained the best held-out validation epoch. NNUE is shared by Iota and Beta; Rho uses the CNN. Metadata records architecture, game and sample counts, role coverage, validation metrics, seed and input data SHA-256.

Rust JSON exports and Python state dictionaries are included. Never confuse the earlier 12-game smoke artifacts with these replacements. Python/Rust parity passed on 40 positions, including every policy logit, with maximum absolute error 6.68e-6.

Dataset SHA-256: `c74f588691d6060b9877393170bce0c1e891896f425e4bda206aa395dc1a68eb`.

The source corpus is a local run artifact; regenerate data with the documented pipeline or include it in an evidence bundle through `python/package_release.py --with-data`.

The exact local pilot training commands used eight epochs, seed 42 and CPU execution:

```powershell
python python/train.py --kind nnue --data runs/training-audited.jsonl --output models/nnue.json --epochs 8 --seed 42 --device cpu
python python/train.py --kind cnn --data runs/training-audited.jsonl --output models/cnn.json --epochs 8 --seed 42 --device cpu
```

Use `evidence/training-audited.jsonl` instead when retraining from the evidence archive. Local dependency/toolchain versions are recorded in `benchmarks/environment.json`; time-limited self-play is not expected to regenerate byte-identical data on different hardware.
