# Benchmarks

Required gauntlet: **50 games per pairing, 300 ms per move**, 25 opening pairs with sides swapped. `benchmarks/openings.json` includes standard start and deterministic legal short openings. They are a reproducible initial suite, not a certified balanced opening book.

```powershell
python python/make_openings.py
python python/benchmark.py --candidate rho --opponents iota beta --output runs/rho-gauntlet
python python/benchmark.py --candidate iota --opponents taflzero opentafl --allow-unaudited --output runs/iota-exploratory
```

External matches are exploratory until their rule compatibility is audited. Missing weights/binaries cause preflight failure. A 50-game run is still provisional while the rules limitations in RULES.md remain. The runner never labels these results final proof.

## Reproducibility

Manifests record engine/model SHA-256 hashes, configuration, Git revision and dirty state, OS/processor, Python, opening suite, seed, resource descriptions, and timing policy. Each game saves moves, side, analysis, transport latency, final FEN, outcome, and termination reason. Separate stderr logs preserve process failures.

Both sides request 300 ms of search. Wall-clock measurement runs from sending `go` to receiving `bestmove`; a default 25-ms IPC allowance is explicit and separately logged. For a strict 300-ms end-to-end run use `--ipc-allowance 0`. Model loading and configured warmups happen before game timing. Pondering is not enabled. Timeouts, illegal moves, and process failures are forfeits; a maximum-ply result is a separate adjudicated draw, not an official rule draw.

The initialization handshake allows up to 120 seconds for slow Python imports/model loading, outside the move clock. Initialization failures abort the runner and can be resumed from saved games. The final campaign records its out-of-clock startup allowance adjustment in `runner-events.json`; no search allowance or completed result changed.

Reports include overall and side-specific W/D/L, score rates, paired-opening bootstrap intervals, latency statistics, and termination counts. Inspect forfeits before interpreting a score as playing strength. Publish CPU/GPU and thread differences, training budgets, and concurrent machine load; they affect conclusions. Re-run ambiguous score differences and avoid extrapolating beyond the tested opponents/hardware.

## External engines

Pinned revisions are in `benchmarks/opponents.lock.json` and `python/opponents.py`:

```powershell
python python/opponents.py --fetch
cargo build --release --no-default-features --manifest-path external/taflzero/Cargo.toml
python python/build_opentafl.py
```

- TaflZero: native UCI, included ONNX weights, exact variant `copenhagen11x11`, CPU ONNX inference. Its batch boundary can overrun short budgets; log those as timeouts rather than crediting a strength win.

- OpenTafl: Java bridge delegates to native Copenhagen rules and upstream alpha-beta, preserving replay history. The bridge uses millisecond cancellation, includes setup in the budget, and sends bestmove before upstream garbage collection. Compatibility build copies fix legacy imports/Base64 calls and make the cancellation flag volatile. Public continuation/horizon options are disabled for subsecond clocks. Native cancellation drain uses a measured 150-ms reserve inside the 300-ms request; ZGC is configured for a bounded 256-MiB heap. Remaining cancellation overruns are logged as timeouts. No evaluator or move/rules logic is substituted. Build manifest records dependency and bridge hashes.

- nicholasg1997/hnefatafl experimental: adapter preserves its network and tree algorithm, replacing only the fixed simulation iterator with a cooperative deadline. Requires upstream-compatible weights and `numba`, `pytorch-lightning` dependencies.

- LuizzK/hnefatafl-engine: timed-loop adapter with upstream network/search and lazy state materialization. Fixed ten-simulation move and policy matched native execution exactly; the measured test improved from 15.4 s to 0.28 s. Requires its checkpoint dictionary. At the pinned revision, its tree expansion uses uniform priors despite computing a neural policy; benchmark that actual behavior, not the README description.

The two Python checkouts have **no pretrained weights** at their pinned revisions. Our explicitly labeled warm starts use 1,000 training games plus 250 validation games through `train_opponent.py`. They are not author-published weights. Random/untrained substitutes are rejected. Focused audits in `benchmarks/luizz-rules-audit.json` and `benchmarks/nicholas-rules-audit.json` show native shieldwall mismatches, so these comparisons remain exploratory.

## Resource isolation and continuation

`--cpu N` pins both players to the same logical CPU. On this i7-1255U, Windows topology identifies CPUs 4–11 as separate equivalent efficiency cores; the final three campaigns use 4, 5 and 6. Referee work and position setup happen outside the search clock. Java uses a 256-MiB heap; PyTorch and ONNX inference use one CPU thread. Other ambient desktop load is recorded as a limitation.

`--resume` checks artifact hashes, the full opening suite and relevant match settings before accepting completed games. JSON records are written atomically. Engines are reused between games and reset through new-game/position commands; after a failure both processes restart. Full stderr is preserved per pairing.

The initial candidates can also overrun under long histories and ambient scheduling. The saved `timing-probe.json` replays one long Beta position on the otherwise free efficiency core: five 300-ms requests returned in 308–317 ms, close to the common 325-ms transport threshold. This diagnostic does not change match scores. Deadline enforcement and forfeits remain part of the published evidence; the current implementation does not guarantee strict real-time response.

Run all fifteen pairings with one command (sequential by default):

```powershell
python python/gauntlet_suite.py --output runs/gauntlets --allow-unaudited
python python/report.py --root runs/gauntlets --output benchmarks/RESULTS.md --require-complete
python python/audit_matches.py --root runs/gauntlets --output runs/replay-audit.json
```

For three parallel campaigns on this machine, add `--cpus 4 5 6` and describe ambient load with `--load-note`. Replay verification checks legality, boards, outcomes and scores against Skallagrim; it is not an independent rules certification. The report separates match scores including forfeits from scores in games completed through play.

Bundle binaries, models, source and documentation using `python python/package_release.py`. Add `--with-data --with-matches` to include the audited training corpus and full final campaign logs. The archive includes checksums for every payload file.

Use `--matches-root runs/gauntlets` for a custom campaign directory. Match evidence also includes the exact locally trained upstream checkpoints and their metadata under `evidence/opponent-artifacts/external/`. Fetch the pinned repositories before copying these checkpoint files into the corresponding `external/` directories. TaflZero's network remains available from its pinned repository.

