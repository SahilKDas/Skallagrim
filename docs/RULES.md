# Copenhagen rules implementation

Authority: [federation rules](https://aagenielsen.dk/copenhagen_rules.php), consulted October 9, 2026, and its [illustrated Copenhagen PDF](https://aagenielsen.dk/Copenhagen_Hnefatafl_11x11.pdf). The PDF clarifies that shieldwalls require a flanking move; filling an inward supporting square is insufficient.

The independent Rust implementation covers the initial arrangement, attacker-first play, rook movement, restricted landing squares, empty-throne passage, king reentry and participation in captures, simultaneous ordinary captures, strict shieldwalls and king immunity to them, four-sided king capture with the throne exception, edge immunity, corner escape, encirclement, repetition, and loss by immobility. `tests/conformance.rs` contains executable cases.

## Operational decisions

- Rank 1 is the first rank in FEN; GUI rank 11 appears at the top. Files are a–k. Moves concatenate endpoints: `f11f9`.
- Repetition counts the full board plus side to move. On its third occurrence, defenders lose. The federation's prose does not prescribe this numerical threshold, so this is an explicit operational interpretation requiring tournament compatibility confirmation.
- Encirclement floods non-attacker squares from all board edges. Every surviving defender, including the king, must be disconnected from the edges to trigger an attacker win.
- Exit forts require the king actually on the edge with a legal king move. The detector repeatedly floods the king's region and erodes defender walls vulnerable to an overapproximation bounded by surviving attacker material and reachable attacker space. Attackers cannot teleport into disconnected pockets or occupy the throne as a supporting piece. A surviving attacker-free enclosure certifies a static fort. This is conservative: some dynamic defenses can produce valid forts that this detector does not recognize.
- A move-limit result in benchmark tooling is **adjudication**, not the federation's impossible-to-finish draw rule. General automatic detection of positions in which neither side can ever finish is not a solved classifier. A referee can record this rule through `adjudicate draw` or GUI key D, followed by evidence review and a written human decision in the referee panel. Qualitative exit forts beyond the automatic sufficient proof can be recorded with `adjudicate fort` or GUI key F and panel confirmation; edge contact and a legal king move are checked. These decisions are explicitly labeled as referee outcomes. No piece-count shortcut is used because it can incorrectly adjudicate playable positions.
- Terminal reason precedence is king capture/corner escape, encirclement, certified fort, repetition, then immobility.
- Captures reset the repetition window because lost material cannot return. The persistent history keeps search clones constant in cost and serializes the current repetition window in chronological order.

## Bounded draw proof

`prove draw [nodes N] [movetime MS]` searches the exact reachable board-and-turn graph, excluding repetition penalties while retaining decisive board outcomes. It reports `ProvenDraw` only if that entire graph closes without a decisive outcome. `FinishReachable` includes a replayable cooperative finishing sequence; it does not establish a forced win. `Unknown` means the time/position budget or cancellation prevented completion. The command does not change the game; a referee can use its evidence before `adjudicate draw`.

This supplies a sound, bounded certificate for a literal reading of "not possible to end," not a universal material-count shortcut or a complete minimax solver. It does not settle the federation's precedence between qualitative draws and repetition. Defaults are 10,000 positions and one second, with hard caps of 100,000 positions and 60 seconds. `stop` cancels asynchronously.

## Certification status

The engine is **not yet certified as fully equivalent to the federation rules**. Remaining work: broader official-position fixtures, confirmation of repetition/draw precedence, and recognition of forts requiring dynamic defensive strategies. Current snapshots identify `copenhagen-static-v3`; the historical 750-game campaign used `copenhagen-static-v2` and is preserved as that version's evidence. Budget-limited draw proofs remain explicitly unknown, and no result is marked as final strength proof.

External engine agreement is useful differential evidence, not a substitute for the rules authority. In particular, historical tafl and Ashton Tablut are not interchangeable with Copenhagen.

The GUI, gauntlets and self-play now use a [separate authoritative Rust referee](REFEREE.md), with independent clocks, position-bound analysis and durable audited rulings.

## Bounded dynamic fort analysis

The dedicated referee supplements static recognition with dynamic-fort-v1 strategy certificates. A separate verifier checks every attacker response and selected defender continuations against a protected king-region invariant. Verified certificates receive an immediate defender ruling; unsupported formations or budget exhaustion stay unknown. Repetition is excluded only in the qualitative invariant graph, not in ordinary games, and an already-terminal repetition loss cannot be overridden. See REFEREE.md for bounds, audit compatibility and strategy previews.
