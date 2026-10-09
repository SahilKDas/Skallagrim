# Skallagrim project notes

Recorded October 9, 2026. These notes capture the agreed project direction. Implementation choices explicitly marked as proposals remain open.

## Required scope

- Implement Copenhagen Hnefatafl on an 11×11 board using the World Tafl Federation's rules.
- Build three candidate bots: **Rho = CNN + MCTS**, **Iota = NNUE + alpha-beta**, and **Beta = NNUE + MCTS**.
- Use Rust and Python.
- Expose a Hnefatafl equivalent of UCI. Use an existing suitable protocol if available; otherwise define one.
- Build a desktop GUI using **tiny-skia**, with detailed visual and interaction polish.
- Benchmark all candidates against TaflZero, nicholasg1997/hnefatafl's experimental branch, LuizzK/hnefatafl-engine, and OpenTafl.
- Support claims of a better bot with **50-game, 300-ms-per-move gauntlets**.
- Train and evaluate competent play as both attackers and defenders. The sides have different objectives and unequal armies; aggregate performance must not conceal weakness on one side.

## Rules authority and correctness

Primary reference: [Rules of Copenhagen Hnefatafl 11×11](https://aagenielsen.dk/copenhagen_rules.php), consulted October 9, 2026; the page identifies its latest update as August 21, 2026.

Use this authority over third-party engine behavior. Implement and test the initial setup, attacker-first turn order, orthogonal movement, ordinary captures, king participation in captures, restricted squares, shieldwalls, corner escape, king capture, exit forts, encirclement, repetition, immobility, and impossible-to-finish draws.

Particular implementation questions to resolve from the authority and documented clarifications:

- The king may return to the throne; other pieces may pass through an empty throne but cannot occupy it.
- Restricted-square hostility depends on the square, piece side, and throne occupancy; the board edge is not generally hostile.
- Shieldwalls can remove defenders alongside the king without removing the king.
- King capture differs next to the throne, and the king cannot be captured on the board edge.
- Exit-fort detection must establish an unbreakable fort, not merely recognize a convenient-looking formation.
- Attacker encirclement is a separate terminal condition.
- The authority penalizes perpetual repetition with a defender loss. Do not substitute an automatic chess-style threefold draw. Precisely document the operational repetition detector before implementation.
- Document how impossible-to-finish draws are detected. A benchmark move limit is a separate adjudication policy and must be recorded as such.

One shared rules core must serve all bots, the GUI, training, and match adjudication. Create focused conformance positions for tricky outcomes and audit external opponents for rule differences before treating matches as comparable.

## Candidate architecture

| Candidate | Required approach | Open design work |
| --- | --- | --- |
| Rho | CNN + MCTS | Board/history encoding, residual tower size, policy encoding, value perspective, exploration, and inference batching |
| Iota | NNUE + alpha-beta | Sparse features, accumulator updates, quantization, training targets, move ordering, and tactical search |
| Beta | NNUE + MCTS | Value calibration, exploration and move priors, and efficient tree/accumulator integration |

NNUE is an evaluation architecture, not a complete search or training recipe. Beta needs an explicit move-prior strategy: a value-only NNUE does not automatically supply Rho's learned policy. Uniform priors, heuristic priors, or a learned policy extension are experimental choices to document and compare.

Proposal: start Iota and Beta with a common NNUE evaluator and checkpoint to isolate search differences, while allowing separate training later. Define all value outputs consistently in terms of attacker/defender or side-to-move perspective; never swap the armies as though their roles were interchangeable.

Training and evaluation must track side-specific performance, sample coverage, and outcomes. Board rotations/reflections may provide augmentation where they preserve the full state and rules; exchanging attacker and defender roles does not.

## Rust / Python responsibilities

Proposed split:

- Rust: authoritative rules/state, move generation, make/unmake, search, neural inference integration, protocol, and GUI.
- Python: CNN/NNUE training, dataset analysis, export, and experiment orchestration.
- Shared versioned formats: positions including repetition-relevant history, moves, training samples, checkpoints/exports, and match logs.

Validate exported model outputs against the training implementation before using them in matches.

## Engine protocol

Existing precedents:

- [OpenTafl engine protocol specification](https://github.com/jslater89/OpenTafl/blob/master/opentafl-engine-protocol.txt)
- [TaflZero UCI-style protocol](https://github.com/sovelin/taflzero#uci-interface)

There are existing tafl protocols, so invention is conditional. Proposal: adopt a TaflZero-compatible UCI-style interface for straightforward benchmarking and provide adapters for OpenTafl and Python opponents. Final protocol selection requires a compatibility review.

The documented contract must cover identification and versioning, readiness, options, new games, variant/rules identity, position and history loading, legal move notation, time-limited search, cancellation, best-move output, analysis output, and orderly shutdown. Keep machine-readable output separate from diagnostics. Support selecting Rho, Iota, or Beta and identifying the loaded weights.

## tiny-skia GUI

tiny-skia is the required rendering foundation; windowing, input, text rendering, and presentation integration still need selection.

Polish requirements: crisp scalable board and piece rendering; distinctive king, attacker, and defender silhouettes; readable coordinates and restricted squares; coherent spacing, typography, and colors; hover/selection states; legal destination hints; last-move and capture feedback; keyboard navigation; high-DPI resizing; move history and replay; side/bot selection; analysis and timing displays; clear outcomes and rule explanations; and responsive asynchronous search with cancellation.

## Benchmark opponents

| Opponent | Source | Integration requirement |
| --- | --- | --- |
| TaflZero | https://github.com/sovelin/taflzero | Pin revision and network; check Copenhagen rules and UCI compatibility |
| nicholasg1997/hnefatafl | https://github.com/nicholasg1997/hnefatafl/tree/experimental | Pin experimental revision and weights; audit rules and build an adapter |
| LuizzK/hnefatafl-engine | https://github.com/LuizzK/hnefatafl-engine | Pin revision and weights; audit rules and build an adapter |
| OpenTafl | https://github.com/jslater89/OpenTafl | Pin revision, evaluation settings, and exact Copenhagen variant; adapt its protocol |

A training pipeline alone is not a trained benchmark opponent. Record whether usable public weights exist; if they do not, establish and report a reproducible training budget. Report unavailable or incompatible opponents explicitly.

## Required gauntlet and proposed conventions

Required: **50 games at 300 ms per move**, with both sides represented.

Proposed interpretation: run a 50-game match for each candidate/opponent pairing, using 25 opening pairs with colors/roles swapped. This gives each candidate 25 attacker games and 25 defender games against each opponent; one round across three candidates and four opponents totals 600 games. Also run candidate-versus-candidate matches. This per-pairing interpretation is a planning assumption, not an additional user requirement.

- Prepare a legal, fixed, versioned opening suite, including the standard start; validate positions and carry repetition history.
- Enforce 300 ms of wall-clock search time per move with a documented common timing boundary and timeout policy. Disable pondering; load models before games. Include inference in search time.
- Pin CPU/GPU hardware, thread limits, memory, inference backend, engine options, revisions, weights, and random seeds. Record each engine's resource allocation and training budget; GPU and CPU comparisons must disclose their differences.
- Let the shared authoritative rules engine adjudicate moves and outcomes. Record illegal moves, timeouts, crashes, draws, and move-limit adjudications separately.
- Publish full logs and overall and side-specific win/draw/loss counts, score rates, uncertainty intervals, and latency statistics.
- A 50-game result is the required evidence unit, not automatic statistical proof. Account for paired openings in uncertainty estimates; replicate ambiguous results before making superiority claims. State which opponents and hardware a claim covers.

## Proposed implementation order

1. Specify and implement the shared Copenhagen rules core with conformance fixtures.
2. Settle the protocol and create a match runner with a simple legal-move baseline.
3. Bring up a tiny-skia GUI early, then refine its polish throughout development.
4. Implement Iota search with a temporary handcrafted evaluator to establish correctness and produce initial data; replace it with the required NNUE.
5. Implement Rho training/inference and MCTS, then Beta using the NNUE with its documented prior strategy.
6. Integrate and audit all four external opponents; run the required gauntlets and iterate using measured results.

## Current status

Only project documentation exists. No implementation, trained candidates, completed gauntlets, or demonstrated superiority is claimed.
