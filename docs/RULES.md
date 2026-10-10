# Copenhagen rules implementation

Authority: [federation rules](https://aagenielsen.dk/copenhagen_rules.php), consulted October 9, 2026, and its [illustrated Copenhagen PDF](https://aagenielsen.dk/Copenhagen_Hnefatafl_11x11.pdf). The PDF clarifies that shieldwalls require a flanking move; filling an inward supporting square is insufficient.

The independent Rust implementation covers the initial arrangement, attacker-first play, rook movement, restricted landing squares, empty-throne passage, king reentry and participation in captures, simultaneous ordinary captures, strict shieldwalls and king immunity to them, four-sided king capture with the throne exception, edge immunity, corner escape, encirclement, repetition, and loss by immobility. `tests/conformance.rs` contains executable cases.

## Operational decisions

- Rank 1 is the first rank in FEN; GUI rank 11 appears at the top. Files are a–k. Moves concatenate endpoints: `f11f9`.
- Repetition counts the full board plus side to move. On its third occurrence, defenders lose. The federation's prose does not prescribe this numerical threshold, so this is an explicit operational interpretation requiring tournament compatibility confirmation.
- Encirclement floods non-attacker squares from all board edges. Every surviving defender, including the king, must be disconnected from the edges to trigger an attacker win.
- Exit forts require the king actually on the edge with a legal king move. The detector repeatedly floods the king's region and erodes defender walls vulnerable to an overapproximation with attackers outside, bounded by surviving material. A surviving attacker-free enclosure certifies a static fort. This is conservative: attacker material and some dynamic defenses can produce valid forts that this detector does not recognize.
- A move-limit result in benchmark tooling is **adjudication**, not the federation's impossible-to-finish draw rule. General automatic detection of positions in which neither side can ever finish is not a solved classifier. A referee can record this rule through `adjudicate draw` or GUI key D. Qualitative exit forts beyond the automatic sufficient proof can be recorded with `adjudicate fort` or GUI key F; edge contact and a legal king move are checked. These decisions are explicitly labeled as referee outcomes. No piece-count shortcut is used because it can incorrectly adjudicate playable positions.
- Terminal reason precedence is king capture/corner escape, encirclement, certified fort, repetition, then immobility.

## Certification status

The engine is **not yet certified as fully equivalent to the federation rules**. Remaining work: broader official-position fixtures, confirmation of repetition policy, more complete exit-fort recognition, and sound impossible-to-finish draw recognition. Match manifests therefore describe `copenhagen-static-v2` and cannot mark results as final strength proof.

External engine agreement is useful differential evidence, not a substitute for the rules authority. In particular, historical tafl and Ashton Tablut are not interchangeable with Copenhagen.
