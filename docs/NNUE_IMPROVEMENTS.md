# Post-campaign NNUE experiments

The October 17 user authorization permits tooling now and **one hour of additional
training after the existing campaign completes**, using existing audited data only.
No new game collection is authorized for this phase. This supersedes the earlier
handoff's prohibition on additional training only within this bounded experiment.
Frozen campaign code, data, weights, clocks and promotion gates remain unchanged.

## Qualification and baseline

First finish the current campaign and pin its earned champions with
`python/pin_campaign_baseline.py`. Rejected candidates are not champions. The
one-hour runner independently requires completed stages, replay audit, verified
promotion decisions and immutable model hashes. Timing-heavy outcomes remain
limited evidence even when all files are replay-valid.

The new referee client records separate bounded request latency and marks a
connection unusable after protocol failure. It never sends a subsequent snapshot
on that connection, extends a referee deadline, or changes a game result. These
changes apply to working source, not to the live frozen campaign.

## Capabilities

`train.py` adds opt-in learning rate, weight decay, patience, minimum improvement,
training-only symmetry augmentation and per-game sample caps. Historical defaults
remain AdamW 0.001 / 0.01, no early stop, no augmentation and all training samples.
New settings and label-manifest hashes participate in durable resume identity.
Validation uses original outcome labels and receives no augmentation or sampling.

`--feature-version v2` adds 45 categorical features to v1's 364 inputs:
repetition count capped at two; minimum king-to-corner Manhattan distance;
adjacent attacker count; and a bitmask of unobstructed corner-reaching rays.
Both runtimes encode the same features. `--migrate-nnue` permits expansion without
changing initial evaluations: existing parameters are copied, new feature and
output contributions start at zero, and extra hidden units retain seeded,
distinct incoming weights so they can learn. The versioned JSON format retains
full v1 compatibility. Capacity comparison uses 64 versus 128 hidden units.

`nnue_dataset_report.py --data DATA --split SPLIT --output REPORT` streams role,
opening, game-length and tactical-proxy coverage. Unknown fort/shieldwall coverage
is explicitly unknown, not certified absent. Reports do not rewrite audited data.

Optional mixed targets require `--search-labels MANIFEST --target-mix FRACTION`.
The manifest contains `data_sha256`, `perspective: "Attacker"`, `training_only:
true`, `engine_path`, `engine_sha256`, `model_path`, `model_sha256`, and `labels`.
Artifacts must still match their hashes. Each label is keyed by
`nnue_experiments.sample_key(row)` and contains the original game ID and a finite
value in [-1,1]. Validation-game labels and missing selected labels are rejected.
No teacher search or replacement labels are generated automatically.

## Bounded study

After tooling verification, run from the workspace:

```powershell
.venv/Scripts/python.exe python/run_nnue_experiments.py --run runs/league-5000-2026-10-13/revision-0002 --output runs/league-5000-2026-10-13/nnue-experiments --verification benchmarks/nnue-improvement-tooling.json
```

Configurations run sequentially from the same pinned champion: control;
learning rate .0003; .0003 with decay .05; .0003 with symmetry; .0003 with v2
features; .0003 with 128 units. All use ten maximum epochs and patience two.
Mixed targets are deliberately skipped until verified teacher labels exist.

The persisted study deadline is never renewed by a restart. Coordinator elapsed
time bounds initialization, validation and checkpoint I/O as well as batches.
At expiry it terminates only its own training child if necessary. Incomplete
epoch updates are discarded; atomic durable checkpoints remain resumable.
Unstarted and incomplete configurations are reported separately from completed
ones. The runner uses Idle priority, CPU 6, one PyTorch thread, half VRAM and
the existing shared 16-GiB cap / 2-GiB reserve; it reuses the existing replay.
An exclusive study lock prevents duplicate coordinators. Verify a stale lock's
PID has exited before removing that one lock for recovery.

## Search measurements and adoption

Use `profile_search.py --baseline BASELINE --positions FROZEN_SUITE --output REPORT`
to record fixed-position trials with original 300-ms requests and 25-ms allowance.
An optional `--engine` selects an experimental release binary while keeping the
pinned weights. Both sides are reached through legal play, with histories retained.
Reports separate engine time, observed transport time, referee request latency,
completed alpha-beta depth, MCTS simulations, fallbacks and diagnostic inference
cost. Eval-rebuild timing is not cumulative search inference time. Older baseline
binaries without timing telemetry report it as unavailable. Diagnostic timeout
draining does not turn a missed deadline into a passing trial.

Profile before optimizing one measured bottleneck at a time. No candidate is
automatically adopted. Correctness, neural parity, timing probes, a two-hour soak,
paired 50-game comparisons, exact role/confidence gates and replay audits remain
required before adoption. Tooling verification is not a speed or strength claim.
