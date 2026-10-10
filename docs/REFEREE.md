# Dedicated Rust referee

The GUI, gauntlet runner and self-play generator use `skallagrim-referee` as a separate child process. It owns the board, captures, side to move, repetition history and result. Bots receive copies for searching and submit proposed moves; only the referee commits them. It shares the tested Copenhagen rules library with the engines, so process separation is not independent rules certification.

Build all binaries with `cargo build --release --bins`. Keep `skallagrim-referee.exe` beside the GUI, or set `SKALLAGRIM_REFEREE` to its absolute path.

## Human panel

Press **J** to open the referee panel. **D/F** open it with a pending draw/fort claim. The board stays visible. Analyze the claim, inspect the status, explored positions and finishing witness, and scroll the evidence with the mouse wheel. Click the reason field to type a decision, select a claim, then click **Confirm human ruling**. Empty reasons are rejected. Use `--referee-name "Alice"` to record the responsible human; the default identifies a local GUI human. Discard clears the pending decision.

Opening the panel cancels and pauses the GUI engine turn, with an explicit audit event. GUI human turns are untimed. Engine turns have a 300-ms search request and a separate 25-ms shared transport allowance: the authoritative hard deadline is 325 ms, including referee acknowledgments and move submission. Match tooling uses its configured search budget plus logged IPC allowance. Self-play uses the existing 1,000-ms transport allowance, aborts on failures, and does not label timeout forfeits as training examples.

## Results and evidence

Captures, corner escape, king capture, encirclement, immobility, third repetition and sufficiently proven static exit forts are resolved by the rules core. Fort analysis first checks the static certificate, then searches for a conservative dynamic strategy. Verified dynamic certificates automatically award a defender win; incomplete searches remain `unknown`. Bounded draw analysis reports `proven` for a completely explored, closed reachable graph with no decisive finish, `finish-reachable` with a cooperative legal witness, or `unknown` on budget exhaustion. A finishing witness is not a forced win. Qualitative cases outside those certificates still require human judgment; no budget exhaustion silently becomes a draw.

Analysis runs on a worker so clocks and moves remain responsive. Evidence belongs to a position revision; a move, reset, undo or timeout prevents stale evidence from attaching to another position. A proven draw certificate is offered to the human as evidence; the result is committed through an explicit human decision. Static forts already recognized by the rules core end the game automatically.

## JSON-line protocol

Requests and responses occupy one UTF-8 line. Supply `id` for response correlation. Responses contain `ok`, `result` or `error`, and the authoritative `state`. Clients attach `revision` to position-sensitive requests; stale revisions are rejected. A rejected move does not change the board or stop its clock. There is no fallback to a local authority when the child fails.

| Command | Fields | Behavior |
| --- | --- | --- |
| `snapshot` | none | Board, history, result, revision, clock, latest analysis and ruling |
| `new` | optional `fen`, `moves` | Atomically validate and start a game/opening |
| `play` | `move`, optional `revision` | Validate, commit, decide captures/terminal rules |
| `clock` | `milliseconds` | Arm the current turn; cannot replace an active clock |
| `pause` | `reason` | Human/controller pause, recorded explicitly |
| `analyze` | `claim`: draw/fort, optional `nodes`, `milliseconds` | Start bounded asynchronous analysis; poll snapshot |
| `adjudicate` | `claim`, `operator`, `reason` | Commit an explicitly identified human decision |
| `undo` | `count` | Rewind the authoritative branch and record the change |
| `forfeit` | `reason`, `evidence` | Record illegal move, failed engine or incompatibility |
| `limit` | `evidence` | Explicit experimental move-limit adjudication |
| `export` | none | Replay with complete audit timeline |
| `validate` | `replay` | Validate a replay and its audit timeline without loading it |
| `quit` | none | Exit the child |

Example:

```json
{"id":1,"command":"snapshot"}
{"id":2,"command":"clock","milliseconds":325,"revision":0}
{"id":3,"command":"play","move":"d1d2","revision":0}
{"id":4,"command":"analyze","claim":"draw","nodes":10000,"milliseconds":500,"revision":1}
```

The deadline uses the referee's monotonic clock from receipt of `clock` through receipt of the complete move line. The timer advances even when no request arrives. Exactly-at-deadline moves lose on time. Clocks are per-move hard deadlines, not cumulative time controls. Draw analysis is limited to 100,000 positions and 10 seconds per request; fort analysis uses the static certificate. Invalid analysis claims are rejected.

## Durable replays

The process automatically writes atomic replay files after changes, analysis results and rulings, before acknowledging a mutation. Default directory: `runs/referee`; override with `--log-dir PATH`. Each new game gets its own file, preserving prior games. A persistence failure stops the authority and reports an error instead of acknowledging an unrecorded ruling.

The GUI's **S** action also exports the full audit. Gauntlet game JSON includes a `referee` section, pins the referee executable hash in the manifest, and rejects incompatible resumes. Self-play writes a compressed `.referee.jsonl.gz` sidecar and individual durable game files beside the corpus.

Each event includes sequence, a reason, position revision, ply, FEN, rules version, resulting outcome and evidence. Rulings identify automatic rules, authoritative clock expiry, a match-controller failure, or a named human with a written reason and available analysis. Undo retains revoked events. Replay loading validates their full timeline, captures, clock evidence, human fields and final branch against the game; it rejects conflicting outcomes and altered board evidence. Finishing witnesses are replayed, static fort certificates are checked, and claimed closed draw graphs are re-explored within a 10-second verification budget; inconclusive re-verification rejects the certificate. Replay validation requires a paused clock. Existing historical replays remain readable without gaining retroactive referee evidence.

## Dynamic fort certificates

Certificate version 1 identifies the root FEN, a fixed protected king region and a graph of FEN nodes with move/successor edges. Every attacker node covers all legal moves; each defender node selects one legal response. All nonterminal nodes preserve an edge king with mobility inside the region and keep attackers outside it. Defender-winning terminal nodes close branches. The bounded search uses greatest-fixed-point elimination to find a closed strategy; a separate verifier regenerates legal moves and checks transitions, coverage, closure and reachable evidence.

Repetition is excluded only from this qualitative fort-invariance graph: a verified fort is awarded immediately under Copenhagen rule 6b, before playing its strategy. Cycles are not a claim that a perpetual sequence could be played without the rule 8 penalty. Already-terminal repetition losses cannot be overridden. The fixed-region condition is sufficient and deliberately conservative; general dynamic forts remain research work. Limits are 100,000 positions, 2,000,000 transitions and ten seconds including verification; exceeding a budget gives unknown.

Search and verification run off the clock-owning thread. Position changes cancel obsolete work; revision checks and clock expiry prevent stale proofs from committing. Replay audit version 2 records dynamic-fort-v1 ruling semantics and verifies certificates again. Version 1 historical audits remain supported. The deterministic rules identifier remains copenhagen-static-v3 because ordinary move rules are unchanged.

The evidence panel lists strategy moves and successor nodes. PgUp/PgDn preview certificate positions on the board; Home restores the authoritative board. Previewing never changes the game. The panel lists the first 100 nodes; all nodes remain available in the replay certificate and board preview.
