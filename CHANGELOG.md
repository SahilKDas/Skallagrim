# Changelog

## Unreleased

- Publish MCTS results before arena cleanup; include prior-worker joining and startup in the caller deadline.
- Shared CLI/GUI response watchdog with exactly one move, cancellation, worker reaping and explicit fallback telemetry.
- Scoped Windows response-thread priority; computation priority and transport limits remain unchanged.
- Add deadline, cleanup-ordering, priority-restoration and neural cancellation regression tests.

## 0.5.0-beta.1

- Dedicated authoritative Rust referee owns moves, repetition, clocks and replay-audited rulings.
- Independently verified bounded dynamic fort certificates, asynchronous analysis and GUI strategy inspection. Unsupported or exhausted proofs remain unknown.
- Role-balanced training league with historical opponents, pinned schedules, difficult starts and separate role promotion gates for Iota and Beta.
- Optional GUI Discord Rich Presence with bounded reconnects, debouncing and clean shutdown. Disabled without an application ID.
- Hardened parsing, search cancellation, replay validation and checkpoint resumes.

This is an experimental beta. Full league training, fresh strength gauntlets and live Discord validation remain separate qualification runs. Historical benchmark charts do not establish superiority. Native manual GUI checks remain untested unless explicitly recorded in the release report.
