# Role-balanced training league

`python/pipeline.py` uses distinct candidate/opponent processes and the authoritative referee. Candidate roles alternate across paired starts; collection requires equal completed candidate-attacker and candidate-defender game counts. Only candidate-turn samples are trained, so historical policy predictions do not become candidate policy targets.

Seeded opponent sampling uses champion 50%, historical promoted checkpoints 30%, current-model self-play 20%. Missing categories are removed and remaining weights normalized. The training pool uses the latest eight historical champions; rejected candidates never enter it. The first generation uses two baseline processes as an explicitly unproven bootstrap.

Every fifth starting pair uses a difficult training position. `benchmarks/league-positions.json` has 25 training and 25 evaluation positions from separate newly generated source games. These legal continuations from standard setup are ranked by nearby attackers, captures and king proximity to an edge. Difficulty is heuristic, not a solved puzzle label. Each records full history, final FEN, rules, seed/source game and a content-derived immutable ID.

```powershell
python python/make_league_positions.py --output runs/new-league-positions.json --seed 1234
python python/pipeline.py --kind nnue --games 1250 --generations 50 --league-positions runs/new-league-positions.json --run runs/new-nnue-league
```

Each generation's `league.json` pins executable/model/metadata/suite hashes, settings, seed and the entire attempt schedule. Changes reject resumes. Interrupted collection restarts deterministically rather than appending duplicates. Logs include opponents, roles, starting provenance and referee events. Truncations and operational forfeits have null targets. Any game visiting a held-out evaluation FEN is excluded in full; replay assembly applies the same filter to older generations.

The minimum remains **1,000 completed training games plus separate game-level validation**, normally 1,250 completed games. Existing storage and attempt limits remain. Sampling proportions describe scheduled attempts; exclusions can alter proportions among completed games.

## Promotion

Each candidate plays 50 games at 300 ms against the champion and up to three latest historical champions. Paired starts give 25 games in each role, with the documented 25-ms transport allowance. A separate 50-game held-out-position gauntlet runs against the champion. CNN gates use Rho; shared NNUE gates must pass through **both Iota and Beta**.

Every gauntlet must complete without operational failures and score at least 50% separately in each role. Standard champion gauntlets also require the lower paired-bootstrap 95% bound above 50%. Held-out and historical gates require the role-specific score floor. These are internal promotion thresholds, not external superiority claims. Completion markers retain every gate summary. Full training and strength gauntlets are separate from smoke checks.
