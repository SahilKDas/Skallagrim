# The actual 5,000-game campaign

One NNUE round, then one CNN round. Each collects 6,250 newly completed referee-audited games: exactly 5,000 training games (2,500 per candidate role) and 1,250 validation games (625 per role). Both warm-start from pinned current models, not Baseline. One candidate attempt per family; rejection retains the old champion.

Run with the workspace environment:

```powershell
.venv/Scripts/python.exe python/run_campaign.py --run runs/league-5000-2026-10-13/revision-0001 --storage-root runs/league-5000-2026-10-13 --publish
```

The controller freezes binaries, Python stage code, initial JSON/PT/metadata, openings and training/evaluation suites under the campaign's runtime directory. It pins external artifacts too. Resume uses committed compressed game transactions, regenerates derived corpora without duplicates and verifies the frozen inputs. If controller code changes, use its frozen copy with explicit --workspace and --run paths. A lock and active-stage record prevent duplicate heavy jobs. Uncommitted games restart; interrupted epochs restart from the last durable epoch, including optimizer and RNG state.

Self-play requests 100 ms, uses exploration 0.15 and the existing 1,000-ms collection transport allowance. Training uses ten epochs, batch 64, unchanged NNUE/CNN shapes, one CPU compute thread and CUDA when available, capped at half its memory. Work runs at idle process priority on logical CPU 6. The shared campaign cap is 16 GiB, replay cap 8 GiB, and free-disk reserve 2 GiB. A random-access SQLite replay stores losslessly compressed full sample rows, avoiding duplicate uncompressed corpora and expensive gzip seeking during training. Budget exhaustion preserves committed data and stops; it never reduces required counts.

The first fifty real completed games contribute to the campaign and provide a measured throughput/ETA. Publish collection progress at 25/50/75/100%, epoch checkpoints and losses, then promotion/evaluation reports. Checkpoints published during training are provisional, not promoted champions. The controller pauses between training/parity and promotion so publication cannot interfere with the match clocks.

Iota and Beta must both pass for a shared NNUE promotion. Rho is gated separately. Each bot plays 50 games at 300 ms against its seed champion and any genuinely promoted historical champions, plus a separate held-out difficult-position gate. Each match has 25 games per role, the unchanged 25-ms IPC allowance, zero operational failures and a minimum 50% score in each role. Champion matches additionally require the paired-bootstrap 95% lower bound above 50%. Export parity is checked on 80 positions before gating. No overrides apply.

The fresh external/mutual suite contains 750 games: fifteen 50-game pairings. It uses new candidate weights even if a candidate failed internal promotion, with that status disclosed. Upstream mismatches, timing failures, and local warm-start provenance remain visible and cannot establish superiority. Full replay audits precede deployment.

Progress, charts, candidate artifacts, hashes and compressed match logs are published under benchmarks/league-5000-2026-10-13. Bulk training samples remain under runs. Deployment changes only evaluators which passed their gates, verifies that the current models have not changed externally, and keeps rejected candidates available for inspection. Actual runs and results are separate from the unit and small integration tests.

The initial launch committed zero games: per-move manifest hashing exhausted a referee clock. Revision 0001 caches that immutable hash and preserves the failed launch alongside the active run under the shared storage cap.
