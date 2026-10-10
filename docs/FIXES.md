# October 9, 2026 bug-fix pass

Current rule identity: `copenhagen-static-v3`. Original benchmark results and their archived binary remain unchanged.

## Fixed

- Coordinate parsing rejects noncanonical ranks and malformed Unicode safely. FEN parsing rejects embedded sentinels/trailing junk. `position startpos` rejects unexpected tokens transactionally.
- Python neural encoding validates the same board restrictions and rejects zero-length/diagonal policy moves instead of silently using negative tensor indexes.
- King capture requires the attacker move to close a neighboring trap; unrelated moves cannot capture an already-surrounded king in an imported position.
- Exit-fort proofs use attacker reachability, material limits, legal throne support and shieldwall vulnerability instead of assuming attacker access to every exterior pocket.
- Search positions share persistent history. Captures discard irreversible earlier repetition states, and long-history destruction is iterative.
- CNN search inference checks cancellation inside convolution rows. CNN dimension validation cannot overflow on malformed model shapes. MCTS respects its depth limit, and explicit depth searches no longer silently inherit 300 ms.
- GUI replay loading checks sides, moves, final boards and terminal decisions. Referee forts require a movable edge king; forged wins and contradictory forfeit winners are rejected. Saving preserves custom starting FENs. Replay navigation cancels pending search results.
- Draw-proof tooling searches a bounded exact reachability graph, returns legal finishing witnesses, and reports exhaustion/cancellation as unknown. It never equates a move cap with an official draw.
- Self-play rejects invalid limits and bounds truncated-game retries while preserving partial output. Training checkpoint writes are atomic, new metadata hashes both model artifacts, and pipeline resumes reject changed training configurations, missing checkpoints and altered hashed artifacts.
- Reports derive timing and hardware from match manifests; they no longer label every future campaign with this laptop's initial settings. Empty summaries are handled, and suite reports default to their campaign directory rather than overwriting historical results. Upstream training validates limits and enforces the game-count gate before initializing CUDA.

## Remaining limitations

The federation's current prose does not prescribe a numerical repetition threshold or its precedence over qualitative draws. Three full-position occurrences remain the documented operational policy. The historical designer discussions and current rules should not be silently combined into a new mandatory variant.

Static proofs are supplemented by bounded, independently verified defensive strategy certificates. Neither mechanism solves every fort formation. A bounded draw proof can return unknown; a cooperative finishing witness does not decide minimax strength. Those positions retain explicit referee support. No universal classifier or federation certification is claimed.

Upstream opponents still have their own rules and timing failures. Repairing their native rules would create modified opponents requiring new provenance and fresh matches. Historical forfeit-heavy charts cannot establish that the fixed engines are stronger.

## Historical evidence

The old engine is preserved at `benchmarks/frozen/target/release/skallagrim.exe`, SHA-256 `470d93c9cb910e0c6f896991cb2ec532858ed41b99ceef7efe2f1a17415634f1`.

```powershell
python python/audit_matches.py --root runs/final-gauntlets --output runs/historical-replay-audit.json --verify-artifacts --artifact-root benchmarks/frozen --engine benchmarks/frozen/target/release/skallagrim.exe
```

Fresh matches obtain their rule identity from the current referee snapshot. `python python/check_deadlines.py` records a separate timing regression probe; it is not strength evidence.

## Verification

62 Rust tests and 34 Python tests passed for the league/presence/fort implementation; the beta release report records fresh checks. Formatting, Clippy and Python undefined-name checks pass. Rust/Python neural parity passes across 80 legal positions, including all policy logits. All 60 timed searches return below 300 ms on the two recorded fixtures. The 1,250-game initial training corpus replays under v3 with no rejected games; the original 750 matches and 35,361 plies verify under their archived v2 binary.

For 0.5.0-beta.1, all 62 Rust and 34 Python tests passed again, with 40-position neural parity (maximum error 6.68e-6). The fresh long soak failed at about 114 minutes on a search-response timeout; a separate 60-search probe included one Beta response above 325 ms. The earlier timing results above are historical, not a guarantee for the beta. Native interactive GUI testing remains untested. See RELEASE.md and benchmarks/release-validation.json.
