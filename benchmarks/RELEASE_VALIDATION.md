# 0.5.0-beta.1 release qualification

**Outcome: partial verification with user overrides (2026-10-10).** Replay loading, proof navigation and the continuous soak have an **override pass**, explicitly accepted by the user. This is a release disposition, not successful verification; the observed results below remain unchanged. The separate deadline-probe failure and other unperformed checks remain unresolved.

| Check | Status | Evidence |
| --- | --- | --- |
| Rust tests | Pass | 62 tests |
| Python tests | Pass | 34 tests with normal temporary-file/process cleanup permissions |
| Formatting, Clippy, Ruff F | Pass | Fresh checks; Ruff from project virtual environment |
| Neural parity | Pass | 40 positions, all 4,840 policy logits, maximum error 6.68e-6 |
| Clean-directory installation | Pass for automated paths | Directory with spaces; developer variables removed; all three bots and referee loaded packaged resources |
| GUI rendering and proof evidence | Pass | Ordinary, replay and 85-node certificate panel renders; proof panel visually inspected |
| Forged/stale fort proofs | Pass | Rust/Python rejection tests |
| Manual GUI walkthrough | User-reported pass | Resizing, new game, legal highlights, move/bot reply and undo; saving displayed a success message |
| Replay loading and proof navigation | Override pass; unverified | User accepted these untested checks; remaining role/bot cycling and window-exit checks are not verified |
| Two-hour continuous process soak | Override pass; observed fail | Timeout after 6,838.531 seconds; 50,629 legal plies, 4,219 replay audits and 633 renders before failure |
| Owned-process shutdown | Pass | All tracked children exited; separate process inventory found no leftovers |
| Working-set observation | Recorded | Peaks: Iota 5.84 MiB, Beta 15.96 MiB, Rho 6.20 MiB, referee 5.78 MiB; paging prevents inference of leak freedom |
| Search deadline probe | Fail against 325-ms requirement | All 60 searches returned, but one Beta response exceeded 325 ms |
| Recovery soak | Separately recorded | See release-recovery.json; does not replace the failed continuous run |
| Full league training, strength gauntlets, live Discord | Not run | Outside this cleanup; no Discord application ID supplied |

Machine-readable evidence: [long soak](release-validation.json), [test checks](release-checks.json), [recovery](release-recovery.json), [timing probe](beta-deadline-regression.json).

The long soak uses repeated 12-ply reset/search/cancel/audit batches, not completed training games. It requests 30-ms searches with a 2,000-ms diagnostic allowance; the normal 300-ms gauntlet allowance remains 25 ms. No limits were relaxed after failure. Timeout cause remains unresolved. Final package extraction is recorded separately in release-package-validation.json.

User acceptance reason, recorded verbatim: "I'm too lazy and im accepting that shit"
