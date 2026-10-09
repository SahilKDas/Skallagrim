# Skallagrim

Skallagrim is a planned Copenhagen Hnefatafl engine and research platform in Rust and Python, with a polished desktop GUI rendered using tiny-skia. Its goal is strong play as both attackers and defenders, backed by reproducible matches against existing open-source engines.

**Status:** Project specification. The rules engine, bots, protocol, training pipeline, GUI, and benchmarks are not implemented yet. Playing strength is a goal to measure, not an established claim.

## Three candidate bots

| Bot | Evaluation | Search |
| --- | --- | --- |
| **Rho** | Convolutional neural network (CNN), policy and value | Monte Carlo Tree Search (MCTS) |
| **Iota** | Efficiently updatable neural network (NNUE) | Alpha-beta |
| **Beta** | Efficiently updatable neural network (NNUE) | MCTS |

All three share the same Copenhagen rules implementation and must play either side competently. Python supports neural training and experiment tooling; Rust supports game logic, search, inference integration, engine communication, and the desktop application.

## Rules and engine communication

The rules authority is the [World Tafl Federation's Copenhagen rules page](https://aagenielsen.dk/copenhagen_rules.php). Correctness includes shieldwalls, throne and corner restrictions, king capture, exit forts, encirclement, repetition, and terminal outcomes.

Skallagrim must expose a documented text protocol equivalent to UCI for tafl. Existing precedents include the [OpenTafl engine protocol](https://github.com/jslater89/OpenTafl/blob/master/opentafl-engine-protocol.txt) and [TaflZero's UCI-style interface](https://github.com/sovelin/taflzero). Adopt or adapt an existing protocol where practical; specify new extensions only where needed.

## GUI

Build a carefully polished tiny-skia desktop interface with a legible board, distinct pieces, clear selection and legal-move feedback, capture and last-move indicators, responsive resizing, accessible controls, move history, and useful engine analysis. Search must run without freezing the interface.

## Measuring strength

Compare Rho, Iota, and Beta with each other and with:

- [TaflZero](https://github.com/sovelin/taflzero)
- [nicholasg1997/hnefatafl, experimental branch](https://github.com/nicholasg1997/hnefatafl/tree/experimental)
- [LuizzK/hnefatafl-engine](https://github.com/LuizzK/hnefatafl-engine)
- [OpenTafl](https://github.com/jslater89/OpenTafl)

The required evidence format is **50-game gauntlets at 300 ms per move**. Use paired starts with sides swapped, publish attacker and defender results separately, and record hardware, engine versions, weights, timing, and complete game logs. Small score differences require uncertainty estimates before claiming a better bot.

See [PROJECT_NOTES.md](PROJECT_NOTES.md) for requirements, benchmark conventions, open design questions, and the proposed implementation order.

## License

[Mozilla Public License 2.0](LICENSE).
