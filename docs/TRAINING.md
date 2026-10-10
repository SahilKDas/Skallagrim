# Training and validation

**Minimum: 1,000 completed training games, plus a separate held-out set.** `train.py` enforces this. A 1,250-game dataset provides a game-level 80/20 split: 1,000 training games and 250 validation games. Truncated attempts carry null targets and are excluded; collection continues until the requested count of completed games is reached. `--smoke` is an explicit exception for pipeline tests, never strength candidates.

Install dependencies in a workspace virtual environment:

```powershell
python -m venv --system-site-packages .venv
.venv/Scripts/python -m pip install -r python/requirements.txt
cargo build --release
```

Generate completed games and train both evaluators:

```powershell
python python/selfplay.py --games 1250 --movetime 100 --output runs/selfplay.jsonl
python python/train.py --data runs/selfplay.jsonl --kind nnue --output models/nnue.json
python python/train.py --data runs/selfplay.jsonl --kind cnn --output models/cnn.json
python python/verify_models.py
```

This baseline generation is a warm start, not the final learning procedure. Subsequent generations must use neural self-play and a substantial recent replay window:

```powershell
python python/pipeline.py --kind cnn --generations 50 --games 1250 --window-games 20000 --movetime 100 --run runs/rho-training
python python/pipeline.py --kind nnue --generations 50 --games 1250 --window-games 20000 --movetime 100 --run runs/nnue-training
```

Each generation persists data, replay, export, PyTorch state, metrics, and completion markers. Completed generations resume without retraining. Promotion uses the [role-balanced league](LEAGUE.md): champion and historical 50-game/300-ms gauntlets, a separate held-out difficult-position gate, role-specific score floors, zero operational failures, and the champion confidence gate. Shared NNUE checkpoints must pass both Iota and Beta. Passing these gates is internal validation, not external strength certification. Iota and Beta can initially share an NNUE checkpoint; their search methods remain different. Beta uses explicit uniform PUCT priors. Rho uses learned policy priors over legal moves.

Training indexes record offsets and loads only each current batch, preserving the game-level split and optimization order. Replay assembly also streams whole games. The default working replay cap is 8 GiB (`--replay-max-gib`); if the budget cannot fit 1,250 completed games, collection/training fails rather than lowering the minimum. The entire learning run has a 16-GiB storage guard (`--max-storage-gib`) and a 2-GiB free-disk reserve. Budget exhaustion stops only the owned stage and its children, preserving completed checkpoints.

Completed self-play and replay corpora are compressed losslessly, verified against their uncompressed SHA-256 before replacing raw files. Selection supports these archives; source IDs are recorded in `replay-selection.json`. Use `--retain-raw` to keep plain files within the same storage budget. `train.py` accepts gzip inputs too, but decompress large archived corpora first for efficient random-access retraining. GPU initialization follows the minimum-data gate, with an allocator cap of half the device memory for successful CUDA training.

## Architecture and targets

- NNUE: 364 sparse binary inputs (three piece-square planes and attacker-to-move), clipped linear hidden layer, linear/tanh attacker-value output. Rust uses incremental accumulators for move, capture, and side changes. Default hidden width is 64.
- CNN: six 11×11 planes (attacker, defender, king, attacker-to-move, restricted squares, current-position repetition count), 3×3 stem, configurable residual blocks, 40-channel 1×1 move-policy head, pooled value head. Default pilot network: 8 channels, 1 residual block. Larger configurations are supported; benchmark their inference cost before scaling.
- Policy indices are `(direction × 10 + distance − 1) × 121 + origin`, directions east/west/north/south. Policy loss and search softmax are normalized over legal moves.
- Value labels are actual terminal outcomes: attacker +1, defender −1, draw 0. Training splits by game rather than position. Best validation epoch is retained. Dataset hashes, counts, both side sample counts, architecture configuration, seed, and losses are saved.
- No attacker/defender exchange augmentation is allowed. Rotations/reflections require transforming the complete state/history/policy; augmentation is not implemented yet.

Python/Rust parity checks compare input encodings, NNUE values, CNN values, and all 4,840 policy logits. Floating-point scalar NNUE is implemented; integer quantization, SIMD, batched CNN inference, and king-relative feature experiments remain optimization/research work.

## Current local bootstrap campaign

The first 12-game checkpoints were smoke artifacts only. A separate 1,250-game campaign used 8-ms baseline search and exploration to exercise the minimum-data pipeline. All games are replay-audited against the corrected strict shieldwall implementation before training. This is minimum-sized bootstrap data, **not a claim of competitive training quality**. The next learning stage is neural self-play at a higher search budget and controlled validation against prior networks and external opponents.

## Upstream checkpoints without published weights

`train_opponent.py` trains upstream-compatible, explicitly labeled Skallagrim warm starts from the same 1,000/250 game split. It uses the upstream encoders and networks: Luizz 16 channels/two blocks for the local pilot, and Nichols' 64-channel/four-layer network. This is a controlled pilot comparison, not a measurement of their authors' trained strength.

```powershell
.venv/Scripts/python python/train_opponent.py --kind luizz --data runs/training-audited.jsonl --output external/luizz/model.pt
.venv/Scripts/python python/train_opponent.py --kind nicholas --data runs/training-audited.jsonl --output external/nicholas/model.ckpt
```

Run the trainings sequentially on memory-constrained machines. Binary feature caches are stored as uint8 and expanded per batch. Metadata includes data hashes, game counts, role coverage, architecture, validation loss, and provenance.
## Resume and interruption safety

Self-play validates positive limits and stops after ten times the requested completed-game count in attempts by default. `--max-attempts` changes that bound; truncated rows remain unlabelled and partial files are preserved on failure. New samples record the referee's rules identity.

Trainer checkpoints use atomic `.pt` replacement, and new metadata hashes both the JSON export and PyTorch state. Pipeline resume verifies those hashes and rejects changed kind, games, move time, epochs, replay-window size or device before modifying the run. Completed generations with missing checkpoint files are rejected. Legacy metadata without artifact hashes can only be checked for file presence and its training-game gate; independently archived hashes remain the stronger evidence for those models.

New self-play runs validate each chosen move through the separate Rust referee and mirror only accepted moves to the search engine. Full audit replays are stored in a compressed `.referee.jsonl.gz` sidecar and individual durable game files; failed or timed-out collection stops without labeling a forfeit as game-theoretic training truth.
