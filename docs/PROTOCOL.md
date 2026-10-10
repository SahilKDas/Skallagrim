# Skallagrim tafl protocol v1

Line-delimited UTF-8 over stdin/stdout, using the existing TaflZero UCI convention. No shell is involved. `info string` carries diagnostic text; machine consumers must ignore unknown informational lines. All commands except readiness/identity/stop cancel and join any prior search before changing state. EOF and `quit` cancel active work.

| Command | Response / behavior |
| --- | --- |
| `uci` | `id`, `option`, `uciok` |
| `isready` | `readyok` |
| `ucinewgame` | Reset standard position and repetition history |
| `setoption name Bot value Rho` | Select `Rho`, `Iota`, `Beta`, or explicitly handcrafted `Baseline` |
| `setoption name NNUEFile value <path>` | Validate and load NNUE JSON; used by Iota and Beta |
| `setoption name CNNFile value <path>` | Validate and load CNN JSON; used by Rho |
| `setoption name Variant value Copenhagen` | The supported variant |
| `position startpos [moves ...]` | Transactionally replay legal moves and their history |
| `position fen <board> <a/d> [moves ...]` | Load FEN; history begins at supplied position, then replay moves |
| `go movetime 300` | Search at most the requested internal budget, then `bestmove <move>` |
| `go nodes 800` | Node-limited search |
| `go depth 4` | Alpha-beta depth limit, also subject to the default 300-ms budget |
| `go infinite` | Search until `stop` or the safety limit |
| `stop` | Cooperatively cancel; join search; exactly one bestmove from the search worker |
| `quit` | Cancel, join, exit |

Moves use `a6b6`, including two-digit ranks. FEN uses `a` attacker, `d` defender, `k` king, decimal empty runs and `/` rank separators. **Rank 1 comes first.** Side to move is `a` or `d`. `0000` means no available move. Errors use `info string error ...`; invalid position/model commands leave the prior state/model intact. Missing neural weights produce an error rather than silently selecting a different evaluator.

Analysis reports `info depth ... score cp ... nodes ... time ... pv ...`. The cp field is a **scaled neural/heuristic value** from the root side's perspective, not calibrated chess centipawns. Neural models themselves return attacker-perspective values in [-1,1]. MCTS also reports JSON visit probabilities in `info string policy ...`.

Skallagrim extensions:

- `snapshot`: `snapshot <JSON>` with FEN, side, terminal outcome, legal moves, full repetition history, captures, and rules identity.
- `play <move>`: apply a validated move, preserving history.
- `legal`: `legal <space-separated moves>`.
- `perft <depth>`: legal path count, stopping at terminal positions.
- `eval`: JSON active sparse features, CNN planes, and loaded neural outputs for parity validation.

Runtime is single-thread search on CPU. Cancellation is checked between search operations; an in-flight CNN evaluation or child expansion can finish after the requested boundary. The external match runner measures wall-clock overruns and records timeouts. Training and GUI consumers do not mistake internal elapsed time for transport timing.

## Referee decisions

`adjudicate draw` records the federation's impossible-to-finish draw decision. `adjudicate fort` records an arbiter-certified exit fort after checking an edge king with a legal move. These commands reject an already finished game. They supplement sufficient automatic proofs; the benchmark runner never silently substitutes them for played outcomes. `Variant` accepts both `Copenhagen` and the TaflZero spelling `copenhagen11x11`.
